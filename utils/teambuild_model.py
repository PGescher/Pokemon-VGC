import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Input: Roster of Pokemon

    Pokemon:
        Base Stast (6D)
        1-2 types (Categorical)
        1-4 moves (given attributes, mixed num/cat?)

Embeddings:
    Pokemon 
    Moves -> Separately and aggregated per Pokemon?

Transformer Encoder:
    No positional encoding -> permutation-invariant input?!?

Predictions Heads:
    Selection score
    Nature
    EV allocation
    Moveset selection

Reinforcement Learning:

"""


# --- Configurable hyperparams ---
MOVE_EMBED_DIM = 32
PKM_EMBED_DIM = 128
TRANSFORMER_LAYERS = 4
TRANSFORMER_HEADS = 8
NATURE_CLASSES = 25
EV_STATS = 6

ENCODED_DIM = 64
MOVE_DIM = 128

# --- Move Config ---
Types = 19
# base_power int
# accuracy float
# max pp int
Categories = 3
# priority int
# effect prob float
# force_switch bool
# self_switch bool
# ignore_evasion bool
# protect bool
# boosts tuple(int x 8)
# self_boosts bool
# heal float
# recoil float
Weather = 5 # Clear, Rain, Sun, Sand, Snow
Terrain = 5 # None, Electric, Grassy, Misty, Psychic
# toogle trockroom bool
# change type bool
# toggle reflect bool
# toggle lightscreen bool
# toggle tailwind bool
Hazard = 3 # None, Stealth rock, Toxic Spikes
Status = 7 # None, Sleep, Burn, Frozen, Paralyzed, Poison, Toxic
# disable bool
Bools = 11
Floats = 4
Ints = 3
Boosts = 8
# 11 + 4 + 3 + 8 = 26


"""
Embeddings:
Embeddings are learned continuous vector representations 
of categorical or discrete features.

This is better than one-hot encoding or manually designed 
features because the embedding space can capture latent 
factors and subtle nuances during training

embedding = learned "meaningful" vector representation of discrete attributes

What does the Embedder do:
Transforms raw input attributed into a fixed-size vector representation.
"""

# Move Embeddings
class MoveEmbedder(nn.Module):
    def __init__(self, n_types=Types, n_categories=Categories, n_weathers=Weather, n_terrains=Terrain, n_hazards=Hazard, n_statuses=Status, move_embed_dim=MOVE_EMBED_DIM, out_dim=64):
        super().__init__()

        # Embeddings
        self.type_emb = nn.Embedding(n_types, move_embed_dim)
        self.cat_emb = nn.Embedding(n_categories, move_embed_dim)
        self.weather_emb = nn.Embedding(n_weathers, move_embed_dim)
        self.terrain_emb = nn.Embedding(n_terrains, move_embed_dim)
        self.hazard_emb = nn.Embedding(n_hazards, move_embed_dim)
        self.status_emb = nn.Embedding(n_statuses, move_embed_dim)

        # MLP for full feature vector
        # Input size:
        #   - 6 embeddings * embed_dim
        total_input_dim = (6 * move_embed_dim + Bools + Floats + Ints + Boosts)

        self.mlp = nn.Sequential(
            nn.Linear(total_input_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self,
                pkm_type,
                category,
                base_power,
                accuracy,
                max_pp,
                priority,
                effect_prob,
                force_switch,
                self_switch,
                ignore_evasion,
                protect,
                boosts,  # (B, N, M, 8)
                self_boosts,
                heal,
                recoil,
                weather_start,
                field_start,
                toggle_trickroom,
                change_type,
                toggle_reflect,
                toggle_lightscreen,
                toggle_tailwind,
                hazard,
                status,
                disable):
        """
        All Move attributes
        t = tensor
        """
        t_type = self.type_emb(pkm_type)
        t_category = self.cat_emb(category)
        t_weather = self.weather_emb(weather_start)
        t_terrain = self.terrain_emb(field_start)
        t_hazard = self.hazard_emb(hazard)
        t_status = self.status_emb(status)

        # Flatten embedding dims
        category_embeddings = [t_type, t_category, t_weather, t_terrain, t_hazard, t_status]
        category_feature = torch.cat(category_embeddings, dim=-1) # (1, 600, 3, 32) = (batch, category_embed * num_pokemon, ?, ?)

        # Normalize Float and Int inputs
        base_features = torch.stack([
            base_power, accuracy, max_pp, priority, effect_prob, heal, recoil
            ], dim=1) # (1, 7, 100, 4) = (Batch, num_base_features, num_pokemon, num_moves)

        # Bool as float32 ?
        bool_features = torch.stack([
            force_switch, self_switch, ignore_evasion, protect, toggle_trickroom, change_type, toggle_reflect, toggle_lightscreen, toggle_tailwind, self_boosts, disable
            ], dim=1).float() # (1, 11, 100, 4) = (Batch, num_bools, num_pokemon, num_moves)

        # boosts = (1, 100, 4, 8) = (Batch, num_pokemon, num_moves, num_boosts)

        #print(t_type.shape, t_category.shape, t_weather.shape, t_terrain.shape, t_hazard.shape, t_status.shape)
        #print(base_features.shape)
        #print(bool_features.shape)
        #print(boosts.shape)
        #print(category_feature.shape)

        # Need to change order and use dim -1 for concat
        base_features = base_features.permute(0, 2, 3, 1)  # (1, 100, 4, 7)
        bool_features = bool_features.permute(0, 2, 3, 1)  # (1, 100, 4, 11)
        #print(base_features.shape)
        #print(bool_features.shape)

        # full input -
        full_input = torch.cat([
            category_feature, base_features, bool_features, boosts.float()
            ], dim=-1) # (B, N, M, total_input_dim)

        B, N, M, _ = full_input.shape

        x = full_input.view(B * N * M, -1)
        out = self.mlp(x)
        return out.view(B, N, M, -1)  # (B, N, M, out_dim)


# Pokemon Embeddings
class PokemonEmbedder(nn.Module):
    def __init__(self, pkm_embed_dim=PKM_EMBED_DIM, n_types=Types, move_embed_dim=MOVE_EMBED_DIM):
        super().__init__()

        self.type_emb = nn.Embedding(n_types, move_embed_dim)
        self.move_embedder = MoveEmbedder()

        # How many dims for a Pokemon?
        # 6 Base Stats, 2 type embed dims, move embed
        in_dim = 6 + 2 * move_embed_dim + move_embed_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, pkm_embed_dim),
            nn.ReLU(),
            nn.Linear(pkm_embed_dim, pkm_embed_dim)
        )

    def forward(self, base_stats, types,
            move_embs):
        # Separate Pokémon-level inputs

        # Embed two Types # ToDo: 1 - 2 Types
        type_1 = self.type_emb(types[:, :, 0])
        type_2 = self.type_emb(types[:, :, 1])
        type_embs = type_1 + type_2 # ToDo: Or Concat

        """
        # Embed moves - (** to unpack dictionary moves)
        move_embs = self.move_embedder(pkm_type, category, base_power, accuracy, max_pp, priority,
            effect_prob, force_switch, self_switch, ignore_evasion, protect,
            boosts, self_boosts, heal, recoil,
            weather_start, field_start, toggle_trickroom, change_type,
            toggle_reflect, toggle_lightscreen, toggle_tailwind,
            hazard, status, disable)
        """
        # Extract only the move-related fields
        #move_inputs = {k: v for k, v in inputs.items() if k not in ['base_stats', 'types']}

        # Pass to embedder
        #move_embs = self.move_embedder(**move_inputs)

        # Aggregate Moves
        move_embs = move_embs.mean(dim=2)
        #print("Pokemon Embedder:")
        #print(f"base_stats: {base_stats.shape}")
        #print(f"type_embs: {type_embs.shape}")
        #print(f"move_embs: {move_embs.shape}")

        # Concat features
        x = torch.cat([base_stats, type_embs, move_embs], dim=-1)

        # mlp to map to vector of pkm_embed_dim
        return self.mlp(x)


"""
Why Transformer?
Why only Encoder?
"""

# Transformer Encoder
class PokemonTransformerEncoder(nn.Module):
    """
    Processes Pokemone Roster
    No Positional Encodings treated as a Set # ToDo: Further understanding of why no pos encoding needed
    """
    def __init__(self, embed_dim=PKM_EMBED_DIM, num_layers=TRANSFORMER_LAYERS, num_heads=TRANSFORMER_HEADS):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x):
        return self.encoder(x)


"""
What are Prediction Heads?
Make Decisions
"""
class SelectionHead(nn.Module):
    def __init__(self, input_dim=PKM_EMBED_DIM):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x: (B, N, d)
        # output: (B, N)
        return self.linear(x).squeeze(-1)


class NatureHead(nn.Module):
    def __init__(self, input_dim=PKM_EMBED_DIM, n_classes=NATURE_CLASSES):
        super().__init__()
        self.linear = nn.Linear(input_dim, n_classes)

    def forward(self, x):
        # x: (B, N, d)
        # output: (B, N, n_classes)
        logits = self.linear(x)
        return logits  # softmax applied externally if needed


class EVHead(nn.Module):
    def __init__(self, input_dim=PKM_EMBED_DIM, n_stats=EV_STATS):
        super().__init__()
        self.linear = nn.Linear(input_dim, n_stats)

    def forward(self, x):
        # x: (B, N, d)
        # output: (B, N, n_stats)
        logits = self.linear(x)
        return logits  # softmax applied externally if needed


class MovesetHead(nn.Module):
    def __init__(self, encoded_dim=ENCODED_DIM, move_dim=MOVE_DIM):
        super().__init__()
        # Interaction of Pokémon and move embeddings
        self.linear = nn.Linear(encoded_dim + move_dim, 1)

    def forward(self, pkm_embeds, move_embeds):
        """
        Inputs:
            pkm_embeds: (B, N, d)  # Pokémon embeddings
            move_embeds: (B, N, M, d_m)  # Moves embeddings

        Output:
            logits: (B, N, M)  # Multi-label sigmoid logits
        """
        B, N, d = pkm_embeds.shape
        M = move_embeds.shape[2]
        d_m = move_embeds.shape[3]

        # Expand Pokémon embeddings to (B,N,M,d)
        pkm_exp = pkm_embeds.unsqueeze(2).expand(-1, -1, M, -1)
        x = torch.cat([pkm_exp, move_embeds], dim=-1)  # (B,N,M,d+d_m)
        x = x.view(-1, d + d_m)                         # (B*N*M, d+d_m)
        logits = self.linear(x).view(B, N, M)           # (B,N,M)

        #print(f"Model Moveset Logits: {logits}")
        return logits
"""
Final Model
"""
class TeamBuilderModel(nn.Module):
    def __init__(self, move_embed_dim=MOVE_EMBED_DIM, pkm_embed_dim=PKM_EMBED_DIM, natures=NATURE_CLASSES, ev_stats=EV_STATS):
        super().__init__()
        self.move_embedder = MoveEmbedder()
        self.pkm_embedder = PokemonEmbedder()
        self.transformer_encoder = PokemonTransformerEncoder()

        self.selection_head = SelectionHead()
        self.nature_head = NatureHead()
        self.ev_head = EVHead()
        self.moveset_head = MovesetHead()

    def forward(self, base_stats, types,
            pkm_type, category, base_power, accuracy, max_pp, priority,
            effect_prob, force_switch, self_switch, ignore_evasion, protect,
            boosts, self_boosts, heal, recoil,
            weather_start, field_start, toggle_trickroom, change_type,
            toggle_reflect, toggle_lightscreen, toggle_tailwind,
            hazard, status, disable):

        move_embeds = self.move_embedder(
                pkm_type,
                category,
                base_power,
                accuracy,
                max_pp,
                priority,
                effect_prob,
                force_switch,
                self_switch,
                ignore_evasion,
                protect,
                boosts,  # (B, N, M, 8)
                self_boosts,
                heal,
                recoil,
                weather_start,
                field_start,
                toggle_trickroom,
                change_type,
                toggle_reflect,
                toggle_lightscreen,
                toggle_tailwind,
                hazard,
                status,
                disable
        )

        # Embed the Pokémon (needs move_embeds)
        pkm_embeds = self.pkm_embedder(base_stats, types, move_embeds)

        # Transformer
        encoded = self.transformer_encoder(pkm_embeds)

        # Prediction heads
        selection_logits = self.selection_head(encoded)
        nature_logits = self.nature_head(encoded)
        ev_logits = self.ev_head(encoded)
        moveset_logits = self.moveset_head(encoded, move_embeds)

        return {
            "selection": selection_logits,
            "nature": nature_logits,
            "ev": ev_logits,
            "moveset": moveset_logits
        }


def encode_roster(roster):
    """
    roster	List[PokemonSpecies]	length = 100
    base_stats	Tuple[int, int, int, int, int, int]	length = 6
    types	List[Type]	length = 1 or 2
    moves	List[Move]	length = 4
    """
    N = len(roster)
    M = 4 # or just 4


    # torch.zeroes is default torch.FloatTensor but i also need Long etc.
    base_stats = torch.zeros(1, N, 6) # 6 stats

    types = torch.zeros(1, N, 2, dtype=torch.long) # 2 possible types

    # Move Attributes
    pkm_type = torch.zeros(1, N, M, dtype=torch.long)
    category = torch.zeros(1, N, M, dtype=torch.long)
    base_power = torch.zeros(1, N, M)
    accuracy = torch.zeros(1, N, M)
    max_pp = torch.zeros(1, N, M)
    priority = torch.zeros(1, N, M, dtype=torch.long)
    effect_prob = torch.zeros(1, N, M)

    force_switch = torch.zeros(1, N, M)
    self_switch = torch.zeros(1, N, M)
    ignore_evasion = torch.zeros(1, N, M)
    protect = torch.zeros(1, N, M)

    boosts = torch.zeros(1, N, M, 8)

    self_boosts = torch.zeros(1, N, M)
    heal = torch.zeros(1, N, M)
    recoil = torch.zeros(1, N, M)

    weather_start = torch.zeros(1, N, M, dtype=torch.long)
    field_start = torch.zeros(1, N, M, dtype=torch.long)

    toggle_trickroom = torch.zeros(1, N, M)
    change_type = torch.zeros(1, N, M)
    toggle_reflect = torch.zeros(1, N, M)
    toggle_lightscreen = torch.zeros(1, N, M)
    toggle_tailwind = torch.zeros(1, N, M)

    hazard = torch.zeros(1, N, M, dtype=torch.long)
    status = torch.zeros(1, N, M, dtype=torch.long)
    disable = torch.zeros(1, N, M, dtype=torch.long)
    disable = torch.zeros(1, N, M, dtype=torch.long)

    for i, pkm in enumerate(roster):
        base_stats[0, i] = torch.tensor(pkm.base_stats)

        # fill types if only 1 type
        types[0, 1, :len(pkm.types)] = torch.tensor([t.value for t in pkm.types])

        # iterate moves of pkm
        for j, move in enumerate(pkm.moves):
            pkm_type[0, i, j] = move.pkm_type.value
            category[0, i, j] = move.category.value
            base_power[0, i, j] = float(move.base_power)
            accuracy[0, i, j] = float(move.accuracy)
            max_pp[0, i, j] = int(move.max_pp)
            priority[0, i, j] = int(move.priority)
            effect_prob[0, i, j] = float(move.effect_prob)

            force_switch[0, i, j] = move.force_switch
            self_switch[0, i, j] = move.self_switch
            ignore_evasion[0, i, j] = move.ignore_evasion
            protect[0, i, j] = move.protect

            boosts[0, i, j] = torch.tensor(move.boosts)

            self_boosts[0, i, j] = move.self_boosts
            heal[0, i, j] = float(move.heal)
            recoil[0, i, j] = float(move.recoil)

            weather_start[0, i, j] = move.weather_start.value
            field_start[0, i, j] = move.field_start.value

            toggle_trickroom[0, i, j] = move.toggle_trickroom
            change_type[0, i, j] = move.change_type
            toggle_reflect[0, i, j] = move.toggle_reflect
            toggle_lightscreen[0, i, j] = move.toggle_lightscreen
            toggle_tailwind[0, i, j] = move.toggle_tailwind

            hazard[0, i, j] = move.hazard.value
            status[0, i, j] = move.status.value
            disable[0, i, j] = move.disable

    return {
        'base_stats': base_stats,
        'types': types,
        'pkm_type': pkm_type,
        'category': category,
        'base_power': base_power,
        'accuracy': accuracy,
        'max_pp': max_pp,
        'priority': priority,
        'effect_prob': effect_prob,
        'force_switch': force_switch,
        'self_switch': self_switch,
        'ignore_evasion': ignore_evasion,
        'protect': protect,
        'boosts': boosts,
        'self_boosts': self_boosts,
        'heal': heal,
        'recoil': recoil,
        'weather_start': weather_start,
        'field_start': field_start,
        'toggle_trickroom': toggle_trickroom,
        'change_type': change_type,
        'toggle_reflect': toggle_reflect,
        'toggle_lightscreen': toggle_lightscreen,
        'toggle_tailwind': toggle_tailwind,
        'hazard': hazard,
        'status': status,
        'disable': disable,
    }

"""
Call like this:

inputs = roster_to_tensors(roster)
outputs = model(
    inputs['base_stats'], inputs['types'],
    inputs['pkm_type'], inputs['category'], inputs['base_power'], inputs['accuracy'], inputs['max_pp'], inputs['priority'],
    inputs['effect_prob'], inputs['force_switch'], inputs['self_switch'], inputs['ignore_evasion'], inputs['protect'],
    inputs['boosts'], inputs['self_boosts'], inputs['heal'], inputs['recoil'],
    inputs['weather_start'], inputs['field_start'], inputs['toggle_trickroom'], inputs['change_type'],
    inputs['toggle_reflect'], inputs['toggle_lightscreen'], inputs['toggle_tailwind'],
    inputs['hazard'], inputs['status'], inputs['disable']
)

"""

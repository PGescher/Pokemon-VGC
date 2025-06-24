from numpy.random import choice, multinomial

from vgc2.agent import TeamBuildPolicy, TeamBuildCommand
from vgc2.battle_engine.modifiers import Nature
from vgc2.meta import Meta, Roster

import torch

from pathlib import Path
from .teambuild_model import TeamBuilderModel

class TransformerTeamBuildPolicy(TeamBuildPolicy):
    def __init__(self):
        self.model = None
        self.last_input = None
        self.last_output = None
        self.last_decision = None

        # ToDo: Init with loaded Model
        current_dir = Path(__file__).resolve().parent
        #model_path = current_dir / 'teambuilder_model_0623_01.pt'
        model_path = current_dir / 'final_teambuilder_model_epoch1000.ckpt'

        self.model = TeamBuilderModel()
        self.model.load_state_dict(torch.load(model_path))

        # model = torch.load(model_path)
        #self.model = torch.jit.load(model_path)
        #print(f"Model loaded: {self.model}")


        self.model.eval()


    def encode_roster(self, roster):
        """
        roster	List[PokemonSpecies]	length = 100
        base_stats	Tuple[int, int, int, int, int, int]	length = 6
        types	List[Type]	length = 1 or 2
        moves	List[Move]	length = 4
        """
        N = len(roster)
        M = 4  # or just 4

        # torch.zeroes is default torch.FloatTensor but i also need Long etc.
        base_stats = torch.zeros(1, N, 6)  # 6 stats

        types = torch.zeros(1, N, 2, dtype=torch.long)  # 2 possible types

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
            'disable': disable
        }



    def decision(self,
                 roster: Roster,
                 meta: Meta | None,
                 max_team_size: int,
                 max_pkm_moves: int,
                 n_active: int) -> TeamBuildCommand:
        #print(f"Max Team Size: {max_team_size}")
        #print(f"Max Pkm Moves: {max_pkm_moves}")



        """
        roster (List): of all available Pokémon to choose from
        meta (Meta): Metadata about the game enviornment
        max_team_size (int): of maximum Pokémon allowed on team
        max_pkm_mover (int): Maximum number of moves each Pokémon can have?
        n_active (int): Number of active Pokémon in battle at a time.

        return: TeamBuildCommand- List of tuples,
        representing a Pokémon and its config for the team.
        """

        # Convert to input tensors for model
        inputs = self.encode_roster(roster)

        # Pass to model
        outputs = self.model(
            inputs['base_stats'], inputs['types'],
            inputs['pkm_type'], inputs['category'], inputs['base_power'], inputs['accuracy'], inputs['max_pp'],
            inputs['priority'],
            inputs['effect_prob'], inputs['force_switch'], inputs['self_switch'], inputs['ignore_evasion'],
            inputs['protect'],
            inputs['boosts'], inputs['self_boosts'], inputs['heal'], inputs['recoil'],
            inputs['weather_start'], inputs['field_start'], inputs['toggle_trickroom'], inputs['change_type'],
            inputs['toggle_reflect'], inputs['toggle_lightscreen'], inputs['toggle_tailwind'],
            inputs['hazard'], inputs['status'], inputs['disable']
        )

        self.last_input = inputs
        self.last_output = outputs

        # Decode outputs into selection
        selection_probs = torch.softmax(outputs['selection'].squeeze(0), dim=0)
        selected_indices = torch.topk(selection_probs, max_team_size).indices.tolist()

        cmds: TeamBuildCommand = []

        for i in selected_indices:
            # Moveset logits shape: (batch, N, M, moves_classes)
            # Assuming moveset logits provide logits per move slot; argmax per move slot to pick moves
            moveset_logits = outputs['moveset'].squeeze(0)[i]  # shape: (M, n_moves_possible)

            #print(f"moveset_logits.shape = {moveset_logits.shape}")
            #print(f"moveset_logits = {moveset_logits}")

            top_scores, top_indices = torch.topk(moveset_logits, max_pkm_moves)

            chosen_moves = top_indices.tolist()
            #print(f"Chosen Moves: {chosen_moves}")
            """
            flat_logits = moveset_logits.flatten()
            top_vals, top_indices = torch.topk(flat_logits, k=max_pkm_moves)

            # Convert flat indices back to (move_slot, move_index)
            move_slots = top_indices // moveset_logits.shape
            move_indices = top_indices % moveset_logits.shape

            chosen_moves = list(zip(move_slots.tolist(), move_indices.tolist()))

            #chosen_moves = moveset_logits.argmax(dim=-1).tolist()
            """

            # EV logits shape: (batch, N, 6)
            ev_logits = outputs['ev'].squeeze(0)[i]
            ev_probs = torch.softmax(ev_logits, dim=0)
            # Here we sample or pick max EV spread
            evs = tuple((ev_probs * 510).int().tolist())

            # Nature logits shape: (batch, N, nature_classes)
            nature_logits = outputs['nature'].squeeze(0)[i]
            nature = Nature(nature_logits.argmax().item())

            # Use perfect IVs (31) as before
            ivs = (31,) * 6

            # Add the command tuple
            #print("Chosen moves (slot_idx, move_idx):", chosen_moves)
            cmds.append((i, evs, ivs, nature, chosen_moves))

        self.last_decision = cmds
        return cmds


"""
All these embeddings can learn, the error signal is backpropagated through:
Transformer, Pokemon encoder, Move encoder, All embedding and MLPS,

We do not need positional encodings as we do not have an order in the Roster.

Raw Roster (list of Pokémon objects)
↓
Embed each Pokémon (via MLP → 128-dim vector, using move encoders, type embeddings, etc.)
↓
Transformer Encoder (contextualizes Pokémon based on whole roster)
↓
Per-Pokémon Output Heads:
    - Select this Pokémon? (logit)
    - Nature choice (25 logits)
    - EV allocation (6-way softmax over 510 budget)
    - Moveset (multi-label over M moves)
↓
Pick N Pokémon for the team (e.g., top 6)
↓
Simulate battles (e.g., against rule-based AI or other learned agents)
↓
Receive reward (win/loss, margin, team synergy score)
↓
Backpropagate reward through model (using Reinforcement Learning: PPO, A2C, etc.)
"""


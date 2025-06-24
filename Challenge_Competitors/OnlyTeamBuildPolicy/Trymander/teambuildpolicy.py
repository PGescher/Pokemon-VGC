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

    """
    Encode roster into tensors:
        Types to int
    decision:
        inputs encoded roster tensors
        model.eval()
        model(inputs)
        topk from selection head
        get corresponding cmds from other heads:
            Nature, EV, Move

    Figure out how learning/training can happen:
        Init the policy and set it on the Competitor, then
        loop and create a new championship/battle each time.
        Loss computation combined, for each head
        Basic Loop with dataloader?
        
    Saving and using the Model:
        torch.save()
        create model object: model = MyModel()
        load model: model.load_state_dict(torch.load("...pt"))
        model.eval

    # Population based learning? (PBT)
        Population: Multiple Policy Models, trained in parallel
        Evaluate: Run Championship with all policies, assign rewards
        Exploit/Explore: Every few epoch rank. Bad ones copy from good ones + mutation/noise
        # ToDo: Research Model Collapse?

    # ToDo: Research Self-Play
    # ToDo: Research Multi-agent RL
    # ToDo: Elo tracking for PBT
    # ToDo: Reasearch Neuroevolution as advanced PBT?
    # ToDo: Research difference between PBT(gradient-based?) and Neuroevolution(fitness&mutation)
    """

    """
    Neuroevolution: evolve best policy without needing a differentiable loss
    Evolve different embedder sizes, number of head or MLP layers?
    We can extract and set weights from the Model.
    instead of .step() we update weights directly based on mutation? No .backward either
    We just copy weights of best ~20%, then fill the 80% with mutations of the top 20%.
    Or use crossover 
    Repeat
    """

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

        # Step 4: For each selected Pokémon, decode moves, nature, evs
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

        #Individual Values perfect stats in attributes
        ivs = (31,) * 6
        #print(f"ivs: {ivs}")
        # Picks 3 Pokémon (ids = choice(len(roster), 3, False))
        ids = choice(len(roster), 3, False)
        #print(f"ids: {ids}")
        cmds: TeamBuildCommand = []
        for i in range(len(ids)):
            # Picks a random subset of its moves.
            n_moves = len(roster[i].moves)
            #print(f"n_moves: {n_moves}")
            # Pick up to max_pkm_moves random unique moves from the available pool.
            moves = list(choice(n_moves, min(max_pkm_moves, n_moves), False))
            #print(f"moves: {moves}")
            # allocates 510 Effort Value (EV) points across 6 stats.
            #multinomial(510, [1/6]*6) evenly distributes points on average, but with randomness.
            evs = tuple(multinomial(510, [1 / 6] * 6, size=1)[0])
            #print(f"evs: {evs}")
            #Randomly selects a Nature (e.g., Adamant, Timid, etc.), which affects stat growth.
            nature = Nature(choice(len(Nature), 1, False))
            #print(f"nature: {nature}")
            #(index, evs, ivs, nature, moves)
            cmds += [(i, evs, ivs, nature, moves)]
            #print(f"cmds: {cmds}")
        return cmds
        
        """

# ToDo: No idea
# Input: Rooster that is randomly generated every time.
# Output: The best Team of 6 Pokemon, with chosen Moves, EV and Nature.

# Option 1: RL with Curriculum. RL agent learns to pick Teams

# Option 2: Evolutionary Strategies: evolve Team-building heuristic
# or parameters. (Weights for scoring Pokemon or Team)

# I would like to use Deep learning to reduce the reliance on manual feature engineering.
# Could i still use this in a RL or ES context?

# Competitive Co-evolution: Multiple Agents learn by competing against each other.
# Or (PBT)


# ToDo: Input/ Data Representation - Encoding Pokemon embeddings
"""
Roster into: tensor of shape (B, N, D):
B = Batch size (Number of roster processed in parallel)
N =  Number of Pokemon in the roster
D = Total pokemon embedding dimension

Encode each Pokemon into a fixed-length feature vector
Feature         Shape       Encoding Strat
# Species       1           Embedding(num_species, d_species)
# Moves         2           Two type Ids -> Embed and sum concat?


#### Ignore EVs and Nature as these are choosen, not available at start
# ToDo: Optional add IV to input encoding
Do we assume perfect IVs? and thus also exlude them? Simplifies a variable

# Ignore these:
# Nature        4           Up to 4 moves -> embed + aggregate
# EVs           6           Normalize or project via Linear(6, d_stats)
# Ivs           1           embedding(num_natures, d_nature)
d_stats = 32
d_nature = 16


d_species = 128
d_type = 32
d_move = 64


D = 128 + 32 + 64 = 224

Summary:
Represent a Pokemon by embeddings:
1. Base stats: Normalize Stat, Vector of size 6 [0,1]
2. Types: Categorical (0-17) embed with learnable vector(if two concat embed or average)
3. Moves: Embed each move into fixed Vector then aggregate to single move embed per pokemon

Single Move Embedding:
Mix of Categorical and continious features:

Aggregate Moves per Pokemon:
Pool into one Vector: Mean pooling or learned attention pooling

Concatenate all Pokemon embeddings:


All these embeddings can learn, the error signal is backpropagated through:
Transformer, Pokemon encoder, Move encoder, All embedding and MLPS,

Questions:

Embedding Models:
What are they?

Full encoding function?

"""


# ToDo: Transformer Architecture:
# Needs Pokemon Embedding
# Transformer input
# Transformer Encoder
# Transformer Heads:
#   - Team Selection: top k
#   - Move Selection:
#   - Nature Selection:
#   - EV Distribution Head
"""
We do not need positional encodings? We do not have an order in the Roster.
Use a Set Transformer or Standard Transformer Encoder.
We only need an Encoder, no decoder?
Optionally Pooling layer?

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


Selecting Pokemon:
class TeamSelector(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.score_head = nn.Linear(embed_dim, 1)         # Select score
        self.nature_head = nn.Linear(embed_dim, 25)       # Softmax
        self.ev_head = nn.Linear(embed_dim, 6)            # EV allocation logits
        self.move_head = nn.Linear(embed_dim, M)          # Multi-label logits over M moves

    def forward(self, encoded_roster):
        select_logits = self.score_head(encoded_roster).squeeze(-1)  # (N,)
        nature_logits = self.nature_head(encoded_roster)             # (N, 25)
        ev_logits = self.ev_head(encoded_roster)                     # (N, 6)
        move_logits = self.move_head(encoded_roster)                 # (N, M)
        return select_logits, nature_logits, ev_logits, move_logits
        
        
# Choose top-K Pokémon (e.g., 6)
top_k_indices = torch.topk(select_logits, k=6).indices
selected_natures = nature_logits[top_k_indices]
selected_evs = ev_logits[top_k_indices]
selected_moves = move_logits[top_k_indices]




"""


# ToDo: Decision function using the trained Transformer Model

# ToDo: Can we use Torch?

import torch
from numpy import argsort
from typing import List

def decision(self,
                 roster: Roster,
                 meta: Meta | None,
                 max_team_size: int,
                 max_pkm_moves: int,
                 n_active: int) -> TeamBuildCommand:
    """
    Use trained Transformer model to choose the best team configuration.
    """
    self.model.eval()

    # Step 1: Encode Roster
    encoded_roster, mask = self.encode_roster_batch([roster])  # (1, N, D), (1, N)
    encoded_roster = torch.tensor(encoded_roster, dtype=torch.float32).to(self.device)
    mask = torch.tensor(mask, dtype=torch.bool).to(self.device)

    # Step 2: Run through Transformer
    with torch.no_grad():
        team_logits, move_logits, nature_logits, evs_out = self.model(encoded_roster, attention_mask=mask)

    # Step 3: Select Top-k Pokémon for team
    team_logits = team_logits[0]  # shape: (N,)
    topk_ids = torch.topk(team_logits, k=max_team_size).indices.cpu().numpy()

    # Step 4: Construct TeamBuildCommand
    cmds: TeamBuildCommand = []
    ivs = (31,) * 6  # perfect IVs (could be configurable)

    for i in topk_ids:
        # Moves: top predicted moves
        moves_scores = move_logits[0, i]  # (num_moves,)
        top_move_ids = torch.topk(moves_scores, k=max_pkm_moves).indices.cpu().numpy()
        available_move_ids = [j for j in top_move_ids if j in roster[i].move_ids][:max_pkm_moves]
        if len(available_move_ids) < max_pkm_moves:
            # Fallback to random from actual available moves
            remaining = [j for j in roster[i].move_ids if j not in available_move_ids]
            available_move_ids += list(choice(remaining, max_pkm_moves - len(available_move_ids), replace=False))

        # Nature
        nature_id = torch.argmax(nature_logits[0, i]).item()
        nature = Nature(nature_id)

        # EVs
        evs = tuple(evs_out[0, i].cpu().round().int().tolist())
        ev_total = sum(evs)
        if ev_total > 510:
            # Normalize down
            evs = tuple(int(e * 510 / ev_total) for e in evs)

        cmds.append((i, evs, ivs, nature, available_move_ids))

    return cmds



# ToDo: Implement weighting of Stats (Heuristic) to control direction?

# ToDo: What broad idea is this approach classified as?
# Genetic Algorithm, MCTS or Reinforcement learning or supervised?
# ToDo: Is there any dataset that can be used?

# ToDo: How do i save and use the trained weights?
# ML model saving mechanisms (pytorch or tensorflow)

# ToDo: Define the Problem Space / Solution Space?

# ToDo: Set a Goal
# Search Rooster for most powerful pokemon? Some Heuristic Archetype?
# This gives me a standout pokemon to build the team around?

# ToDo: Steps
#
# Last Step: Nature Selection - Determine 1st and 2nd main Stat
# and Boost Main stat but such that the secondary stat is not impeded
# ToDo: How to Determine most important Stat?

# ToDo: Decide on an Optimization Strategy
# Heuristic Search - self defined Fitness function (Genetic Algorithms, simulated annealing)
# Monte Carlo Tree Search (MCTS) -

# ToDo: Learn function


# ToDo: Mutation function?



# ToDo: Pokemon Score function (Weights * Features)
# Base Stats (6)
# Pokemon Type 19
# Moves Summary of all moves 22


"""
Notes:
Do not learn identities, learn features.
I need to implement learning so that it learns what features of a pokemon are desirable.

What to learn?
Encode pokemon? 
- Weights of Pokemon features to calculate score of a pokemon? Base Stats (six stats)
- Types of pokemon (Fixed order? A score for each? What are all the possible types?)
- Move Features - (What features do moves have)?
- Aggregate the features of all moves?
    -Access moves attributes programmatically move.accurarcy etc..

How to do Synergy?
Feature Score of Team thus far?

First Pick best overall pokemon
Now update the context with the average of the selected pokemons
Now we recaluclate the score for each pokemon adding the context we computed.
Now we again select the pokemon with the highest score.
And repeat - Select Pokemon -> Compute average context -> Compute pokemon score -> repeat

I need to integrate the choice of Moves from the Moves of the pokemon into the team building.
In addition the EVs point distribution as well as the Nature choice

Do i consider all possible combinations of moves for the pokemon choice or do i still just calculate the average score 
for the moves and then select the moves for the pokemon in a next step?

Will the EVs or the Nature be considered after the Pokemon selection or also during?
Ideally the Pokemons chosen without EV or Nature would already be the best Pokemon and 
thus remain the best with EV and nature

Try a set of EV/Nature sets for each pokemon and keep the best score pokemon for the further selection.
This can be hard do to a limit in EV points.

How do the EV points and the Nature actually affect the pokemon?

Maybe just use heuristic, assuming a pokemon will be really strong in a single stat, its main stat and weight that higher?
Then select the most appropriate nature? How would that work?

What is the best strategy for EV and Nature?

Nature should be selected based on highest and lowest stat.

I should compute all move combinations and calculate the most important stats. Primary / Secondary ... Stat
Then select a Nature based on the highest and lowest stat
Then Score each pokemon with the resulting stat nature and move combination. 

On the first pokemon selection i should pick the pokemon with the best score.

On the second pick i would somehow like to consider the already selected pokemons for the creating of the pokemons 
ideal pokemon for the score calculation.
So the team weight needs to be considered on the pokemon move and stat calculation.
This means the pokemon move and stat calculation needs to be redone after every selected pokemon.
....
Is there some clever set of weights i can learn to just pick the best pokemon combinations?

"""

"""
Have a function assign each move a primary and secondary attribute.

(pokemon_base_stats, move_features_list) → EV distribution (6 floats summing to 510)

Which stats are valuable based on move types (e.g. PHYSICAL → ATK, SPECIAL → SpAtk)
When Speed is critical (e.g. if you have high-priority or fast moves)
When HP, Def, SpDef should be boosted (e.g. for setup moves or tanks).
How to trade off stats within the EV budget.

"""


"""
Design a policy that builds a team of 6 Pokémon from a roster by:

Choosing which 6 Pokémon to include

Choosing a moveset (4 moves) per Pokémon

Choosing EV distribution per Pokémon

Choosing a Nature per Pokémon
"""



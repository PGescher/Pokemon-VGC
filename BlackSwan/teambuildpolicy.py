from numpy.random import choice, multinomial

from vgc2.agent import TeamBuildPolicy, TeamBuildCommand
from vgc2.battle_engine.modifiers import Nature
from vgc2.meta import Meta, Roster

class RandomTeamBuildPolicy(TeamBuildPolicy):
    """
    random team builder.
    """

    """
    In the game random moves are generated
    Pokemons are generated with a set from those random moves
    EVs are Effort Values: Max 510 Total and single cant exceed 252
    
    Potential Goals:
    Type Coverage,
    Stat balance,
    Role diversity,
    Synergy,
    Move utility
    Meta-awareness
    
    """

    def decision(self,
                 roster: Roster,
                 meta: Meta | None,
                 max_team_size: int,
                 max_pkm_moves: int,
                 n_active: int) -> TeamBuildCommand:
        """
        roster (List): of all available Pokémon to choose from
        meta (Meta): Metadata about the game enviornment
        max_team_size (int): of maximum Pokémon allowed on team
        max_pkm_mover (int): Maximum number of moves each Pokémon can have?
        n_active (int): Number of active Pokémon in battle at a time.

        return: TeamBuildCommand- List of tuples,
        representing a Pokémon and its config for the team.
        """

        #Individual Values perfect stats in attributes
        ivs = (31,) * 6
        print(f"ivs: {ivs}")
        # Picks 3 Pokémon (ids = choice(len(roster), 3, False))
        ids = choice(len(roster), 3, False)
        print(f"ids: {ids}")
        cmds: TeamBuildCommand = []
        for i in range(len(ids)):
            # Picks a random subset of its moves.
            n_moves = len(roster[i].moves)
            print(f"n_moves: {n_moves}")
            # Pick up to max_pkm_moves random unique moves from the available pool.
            moves = list(choice(n_moves, min(max_pkm_moves, n_moves), False))
            print(f"moves: {moves}")
            # allocates 510 Effort Value (EV) points across 6 stats.
            #multinomial(510, [1/6]*6) evenly distributes points on average, but with randomness.
            evs = tuple(multinomial(510, [1 / 6] * 6, size=1)[0])
            print(f"evs: {evs}")
            #Randomly selects a Nature (e.g., Adamant, Timid, etc.), which affects stat growth.
            nature = Nature(choice(len(Nature), 1, False))
            print(f"nature: {nature}")
            #(index, evs, ivs, nature, moves)
            cmds += [(i, evs, ivs, nature, moves)]
            print(f"cmds: {cmds}")
        return cmds


#ToDo: Start Simple? Learn EV distribution based on initial Stats somehow?

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
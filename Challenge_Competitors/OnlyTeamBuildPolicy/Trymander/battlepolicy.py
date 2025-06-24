from itertools import product
from math import prod
from random import sample
from typing import Optional

from numpy import argmax
from numpy.random import choice

from vgc2.agent import BattlePolicy
from vgc2.battle_engine import State, BattleCommand, calculate_damage, BattleRuleParam, BattlingTeam, BattlingPokemon, \
    BattlingMove, TeamView
from vgc2.util.forward import copy_state, forward
from vgc2.util.rng import ZERO_RNG, ONE_RNG


# RandomBattlePolicy

class RandomBattlePolicy(BattlePolicy):
    """
    Policy that selects moves and switches randomly. Tailored for single and double battles.
    """

    def __init__(self,
                 switch_prob: float = .15):
        self.switch_prob = switch_prob

    def decision(self,
                 state: State,
                 opp_view: Optional[TeamView] = None) -> list[BattleCommand]:
        team = state.sides[0].team
        n_switches = len(team.reserve)
        n_targets = len(state.sides[1].team.active)
        cmds: list[BattleCommand] = []
        for pkm in team.active:
            n_moves = len(pkm.battling_moves)
            switch_prob = 0 if n_switches == 0 else self.switch_prob
            action = choice(n_moves + 1, p=[switch_prob] + [(1. - switch_prob) / n_moves] * n_moves) - 1
            if action >= 0:
                target = choice(n_targets, p=[1 / n_targets] * n_targets)
            else:
                target = choice(n_switches, p=[1 / n_switches] * n_switches)
            cmds += [(action, target)]
        return cmds
# GreedyBattlePolicy

def greedy_single_battle_decision(params: BattleRuleParam,
                                  state: State) -> list[BattleCommand]:
    attacker, defender = state.sides[0].team.active[0], state.sides[1].team.active[0]
    outcomes = [calculate_damage(params, 0, move.constants, state, attacker, defender)
                if move.pp > 0 and not move.disabled else 0 for move in attacker.battling_moves]
    return [(int(argmax(outcomes)), 0) if outcomes else (0, 0)]


def greedy_double_battle_decision(params: BattleRuleParam,
                                  state: State) -> list[BattleCommand]:
    attackers, defenders = state.sides[0].team.active, state.sides[1].team.active
    strategies = []
    for sources in product(list(range(len(attackers[0].battling_moves))),
                           list(range(len(attackers[1].battling_moves))) if len(attackers) > 1 else []):
        for targets in product(list(range(len(defenders))), list(range(len(defenders)))):
            damage, ko, hp = 0, 0, [d.hp for d in defenders]
            for i, (source, target) in enumerate(zip(sources, targets)):
                attacker, defender, move = attackers[i], defenders[target], attackers[i].battling_moves[source]
                if move.pp == 0 or move.disabled:
                    continue
                new_hp = max(0, hp[target] - calculate_damage(params, 0, move.constants, state, attacker, defender))
                damage += hp[target] - new_hp
                ko += int(new_hp == 0)
                hp[target] = new_hp
            strategies += [(ko, damage, sources, targets)]
    if len(strategies) == 0:
        return [(choice(len(a.battling_moves)), choice(len(defenders))) for a in attackers]
    best = max(strategies, key=lambda x: 1000 * x[0] + x[1])
    return list(zip(best[2], best[3]))


class GreedyBattlePolicy(BattlePolicy):
    """
    Greedy strategy that prioritizes KOs and damage output with only one turn lookahead. Performs no switches.
    """

    def __init__(self,
                 params: BattleRuleParam = BattleRuleParam()):
        self.params = params

    def decision(self,
                 state: State,
                 opp_view: Optional[TeamView] = None) -> list[BattleCommand]:
        n_active_0, n_active_1 = len(state.sides[0].team.active), len(state.sides[1].team.active)
        match max(n_active_0, n_active_1):
            case 1:
                return greedy_single_battle_decision(self.params, state)
            case 2:
                return greedy_double_battle_decision(self.params, state)


# TreeSearchBattlePolicy

def get_actions(team: tuple[BattlingTeam, BattlingTeam]) -> list[list[BattleCommand]]:
    attackers = team[0].active
    move_targets = [i for i in range(len(team[1].active))]
    switch_targets = [i for i, p in enumerate(team[0].reserve) if p.hp > 0]
    commands = []
    for attacker in attackers:
        moves = [i for i, m in enumerate(attacker.battling_moves) if m.pp > 0 and not m.disabled]
        commands += [list(product(moves, move_targets)) + list(product([-1], switch_targets))]
    return list(product(*commands))


def _deduce_moves(pokemon: BattlingPokemon,
                  max_moves: int):
    n_moves = len(pokemon.battling_moves)
    if n_moves < max_moves:
        ids = [m.constants.id for m in pokemon.battling_moves]
        moves = [m for m in pokemon.constants.species.moves if m.id not in ids]
        pokemon.battling_moves += [BattlingMove(m) for m in sample(moves, max_moves - n_moves)]  # ignoring meta


def deduce_state(state: State,
                 opp_team_view: TeamView,
                 max_moves: int) -> State:
    _state = copy_state(state)
    opp_team = _state.sides[1].team
    # randomly assume reserve of opponent
    current_pokemon = len(opp_team.active + opp_team.reserve)
    total_pokemon = len(opp_team_view.members)
    if current_pokemon < total_pokemon:
        ids = [p.constants.species.id for p in opp_team.active + opp_team.reserve]
        pokemon = [p for p in opp_team_view.members if p.species.id not in ids]
        opp_team.reserve += [BattlingPokemon(p) for p in sample(pokemon, total_pokemon - current_pokemon)]
    # randomly set hidden moves of opponent
    for p in opp_team.active + opp_team.reserve:
        _deduce_moves(p, max_moves)
    return _state


def eval_state(state: State) -> float:
    my_team = state.sides[0].team
    my_hp = sum(p.hp / p.constants.stats[0] for p in my_team.active + my_team.reserve)
    opp_team = state.sides[1].team
    opp_hp = sum(p.hp / p.constants.stats[0] for p in opp_team.active + opp_team.reserve)
    return my_hp - 3 * opp_hp + 3. * (len(opp_team.active) + len(opp_team.reserve))

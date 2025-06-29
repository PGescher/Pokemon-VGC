from vgc2.agent import BattlePolicy, SelectionPolicy, TeamBuildPolicy
#from vgc2.agent.battle import RandomBattlePolicy
#from vgc2.agent.selection import RandomSelectionPolicy
#from vgc2.agent.teambuild import RandomTeamBuildPolicy
from vgc2.competition import Competitor

from Trymander.battlepolicy import GreedyBattlePolicy
from Trymander.selectionpolicy import RandomSelectionPolicy
from Trymander.teambuildpolicy import TransformerTeamBuildPolicy

class TrymanderCompetitor(Competitor):

    def __init__(self, name: str = "Trymander"):
        self.__name = name
        self.__battle_policy = GreedyBattlePolicy()
        self.__selection_policy = RandomSelectionPolicy()
        self.__team_build_policy = TransformerTeamBuildPolicy()

    @property
    def battle_policy(self) -> BattlePolicy | None:
        return self.__battle_policy

    @property
    def selection_policy(self) -> SelectionPolicy | None:
        return self.__selection_policy

    @property
    def team_build_policy(self) -> TeamBuildPolicy | None:
        return self.__team_build_policy

    @property
    def name(self) -> str:
        return self.__name

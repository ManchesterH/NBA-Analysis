from typing import List
from src.player import Player
from src.config import PARAMS

class Team:
    def __init__(self, name: str, market_size_factor: float = 1.0):
        self.name = name
        self.roster: List[Player] = []
        self.brand_value: float = 100.0
        self.market_size_factor = market_size_factor
        self.cash: float = 0.0
        self.franchise_value: float = 0.0
        
    def add_player(self, player: Player):
        self.roster.append(player)
        
    def remove_player(self, player_name: str):
        self.roster = [p for p in self.roster if p.name != player_name]
        
    @property
    def total_payroll(self) -> float:
        return sum(p.salary for p in self.roster)
    
    @property
    def team_ps_base(self) -> float:
        return sum(p.ps_base for p in self.roster)
        
    @property
    def team_ps_adjusted(self) -> float:
        return sum(p.ps_adjusted for p in self.roster)
        
    @property
    def active_roster_popularity(self) -> float:
        return sum(p.popularity_score for p in self.roster)
        
    def update_brand_value(self, playoff_success_level: int):
        decay = PARAMS.BRAND_VALUE_DECAY
        boost = PARAMS.BRAND_VALUE_PLAYOFF_BOOST * playoff_success_level
        self.brand_value = self.brand_value * (1 - decay) + boost

    def apply_injuries(self, load_factor: float = 1.0):
        for player in self.roster:
            player.check_for_injury(load_factor)
            player.recover()

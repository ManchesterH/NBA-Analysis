from dataclasses import dataclass, field
import random
import math
from typing import Optional
from src.config import PARAMS
from src.utils import calculate_popularity_score, calculate_potential_score

@dataclass
class Player:
    name: str
    age: int
    per: float
    ewa: float
    rank_per: int
    years_top_30: int
    salary: float
    is_injured: bool = False
    injury_duration: int = 0
    _popularity_raw: Optional[float] = None
    _potential_raw: Optional[float] = None
    
    def __post_init__(self):
        self.update_metrics()

    def update_metrics(self):
        self._popularity_raw = calculate_popularity_score(self.rank_per, self.years_top_30)
        self._potential_raw = calculate_potential_score(self.age)

    @property
    def popularity_score(self) -> float:
        if self._popularity_raw is None:
            self.update_metrics()
        return self._popularity_raw

    @property
    def potential_score(self) -> float:
        if self._potential_raw is None:
            self.update_metrics()
        return self._potential_raw

    @property
    def ps_base(self) -> float:
        p_score = (
            PARAMS.WEIGHT_PER * self.per +
            PARAMS.WEIGHT_POPULARITY * self.popularity_score +
            PARAMS.WEIGHT_EWA * self.ewa +
            PARAMS.WEIGHT_POTENTIAL * self.potential_score
        )
        return p_score

    @property
    def ps_adjusted(self) -> float:
        if self.is_injured:
            return 0.0
        return self.ps_base

    def check_for_injury(self, load_factor: float = 1.0) -> bool:
        prob_injury = 1 - math.exp(-PARAMS.INJURY_RATE_LAMBDA * load_factor)
        if random.random() < prob_injury:
            self.is_injured = True
            self.injury_duration = random.randint(10, 30)
            return True
        return False

    def recover(self):
        if self.is_injured:
            self.injury_duration -= 1
            if self.injury_duration <= 0:
                self.is_injured = False
                self.injury_duration = 0

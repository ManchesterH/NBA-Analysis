import numpy as np
from typing import Tuple, Dict
from src.config import PARAMS
from src.utils import logistic_demand
import math

class TicketOptimizer:
    @staticmethod
    def expected_revenue(price: float, capacity: int, p_star: float, lambd: float) -> float:
        d = logistic_demand(price, capacity, p_star, lambd)
        return price * d

    @staticmethod
    def optimize_price(capacity: int, p_star: float, lambd: float) -> float:
        best_p = 0.0
        max_rev = -1.0
        search_range = np.linspace(p_star * 0.5, p_star * 2.0, 100)
        for p in search_range:
            rev = TicketOptimizer.expected_revenue(p, capacity, p_star, lambd)
            if rev > max_rev:
                max_rev = rev
                best_p = p
        return best_p

class InjuryMDP:
    def __init__(self, horizon_games: int = 82):
        self.horizon = horizon_games
        self.states: Dict[Tuple[int, int], float] = {}
        self.policy: Dict[Tuple[int, int], int] = {}
        self.win_value = 1.0
        self.rest_recovery = 2
        self.play_fatigue = 1
        self.max_fatigue = 10
        self.injury_penalty = -50.0
    
    def solve(self):
        V_next = np.zeros((self.max_fatigue + 1, 2))
        for t in range(self.horizon - 1, -1, -1):
            V_curr = np.zeros((self.max_fatigue + 1, 2))
            for f in range(self.max_fatigue + 1):
                V_curr[f, 0] = self.injury_penalty + V_next[0, 0]
                f_next_rest = max(0, f - self.rest_recovery)
                val_rest = 0 + V_next[f_next_rest, 1]
                lambda_game = 0.05 
                prob_injury = 1 - math.exp(-lambda_game * (f + 1))
                f_next_play = min(self.max_fatigue, f + self.play_fatigue)
                val_play = (1 - prob_injury) * (self.win_value + V_next[f_next_play, 1]) + \
                           prob_injury * (self.injury_penalty + V_next[0, 0])
                if val_play > val_rest:
                    V_curr[f, 1] = val_play
                    if t == 0: self.policy[(f, 1)] = 1
                else:
                    V_curr[f, 1] = val_rest
                    if t == 0: self.policy[(f, 1)] = 0
            V_next = V_curr
            
    def get_optimal_action(self, fatigue: int, is_healthy: bool) -> str:
        if not is_healthy:
            return "Heal"
        f = min(self.max_fatigue, max(0, int(fatigue)))
        action = self.policy.get((f, 1), 1)
        return "Play" if action == 1 else "Rest"

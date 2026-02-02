import math
import numpy as np

def calculate_popularity_score(rank_per: int, years_top_30: int) -> float:
    if rank_per < 1:
        rank_per = 1
    term1 = math.exp(-rank_per / 110.0)
    term2 = math.log(1 + years_top_30)
    return term1 * term2

def calculate_potential_score(age: float) -> float:
    if age < 18:
        t = 0.0
    else:
        t = age - 18
    term1 = 0.05 * (t**2) * math.exp(-t / 1.5)
    term2 = 0.0002 * (t**5) * math.exp(-t / 2.0)
    return 50 * (term1 + term2)

def logistic_demand(price: float, capacity: int, p_star: float, lambd: float) -> float:
    try:
        exponent = lambd * (price - p_star)
        if exponent > 700:
            return 0.0
        denom = 1 + math.exp(exponent)
        return capacity / denom
    except OverflowError:
        return 0.0

def pythagorean_win_prob(ps_team: float, ps_opponent: float) -> float:
    ps_team_sq = ps_team ** 2
    ps_opp_sq = ps_opponent ** 2
    if ps_team_sq + ps_opp_sq == 0:
        return 0.5
    return ps_team_sq / (ps_team_sq + ps_opp_sq)

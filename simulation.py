import copy
import random
import numpy as np
from typing import List, Dict
from src.team import Team
from src.config import PARAMS
from src.financials import RevenueModel, CostModel
from src.competition import PlayoffSimulator

class SimulationEngine:
    def __init__(self, initial_team: Team, years: int = 5):
        self.initial_team = initial_team
        self.years = years
        
    def run_single_path(self) -> Dict:
        team = copy.deepcopy(self.initial_team)
        path_data = {
            "years": [],
            "revenue": [],
            "cost": [],
            "profit": [],
            "franchise_value": [],
            "playoff_rounds": []
        }
        current_value = 3000.0
        for t in range(self.years):
            for p in team.roster:
                p.age += 1
            team.apply_injuries()
            regular_ticket_rev = RevenueModel.calculate_regular_ticket_revenue(team, t)
            rounds_won = PlayoffSimulator.simulate_playoff_run(team)
            playoff_ticket_rev = RevenueModel.calculate_playoff_ticket_revenue(team, rounds_won)
            team.update_brand_value(rounds_won)
            merch_rev = RevenueModel.calculate_merchandise_revenue(team, t)
            sponsor_rev = RevenueModel.calculate_sponsorship_revenue(team)
            broadcast_rev = RevenueModel.calculate_broadcast_revenue(team, t)
            total_revenue = regular_ticket_rev + playoff_ticket_rev + merch_rev + sponsor_rev + broadcast_rev
            total_cost = CostModel.calculate_total_cost(team, t)
            profit = total_revenue - total_cost
            current_value += profit
            path_data["years"].append(t+1)
            path_data["revenue"].append(total_revenue)
            path_data["cost"].append(total_cost)
            path_data["profit"].append(profit)
            path_data["franchise_value"].append(current_value)
            path_data["playoff_rounds"].append(rounds_won)
        return path_data

    def run_monte_carlo(self, num_simulations: int = 100) -> Dict:
        results = {
            "final_values": [],
            "avg_profit": []
        }
        for _ in range(num_simulations):
            path = self.run_single_path()
            results["final_values"].append(path["franchise_value"][-1])
            results["avg_profit"].append(np.mean(path["profit"]))
        return results

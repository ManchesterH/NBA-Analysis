import math
from src.config import PARAMS
from src.team import Team

class RevenueModel:
    @staticmethod
    def calculate_regular_ticket_revenue(team: Team, year_index: int, average_price: float = None) -> float:
        if average_price is None:
            average_price = PARAMS.AVERAGE_TICKET_PRICE_BASE
        t0 = 2
        growth_factor = 1.0 / (1.0 + math.exp(-PARAMS.TICKET_GROWTH_RATE * (year_index - t0)))
        games_home = 41
        utilization = min(1.0, 0.8 + (team.team_ps_adjusted / 500.0) * 0.2)
        revenue = PARAMS.STADIUM_CAPACITY * utilization * average_price * games_home / 1_000_000.0
        return revenue

    @staticmethod
    def calculate_playoff_ticket_revenue(team: Team, rounds_reached: int) -> float:
        revenue = 0.0
        prices = [200, 300, 450, 700]
        games_expected = 3
        for r in range(rounds_reached):
            rev_round = (PARAMS.STADIUM_CAPACITY * games_expected * prices[r]) / 1_000_000.0
            revenue += rev_round
        return revenue

    @staticmethod
    def calculate_merchandise_revenue(team: Team, year_index: int) -> float:
        inflation_factor = (1 + PARAMS.INFLATION_RATE) ** year_index
        base_component = PARAMS.MERCH_BASE * inflation_factor
        gamma = 19.4
        pop_component = gamma * team.active_roster_popularity
        return base_component + pop_component

    @staticmethod
    def calculate_sponsorship_revenue(team: Team) -> float:
        beta = 0.2
        return PARAMS.SPONSORSHIP_FIXED + beta * team.brand_value

    @staticmethod
    def calculate_broadcast_revenue(team: Team, year_index: int) -> float:
        national = PARAMS.BROADCAST_NATIONAL
        league_avg_ps = 400.0 
        ps_factor = team.team_ps_adjusted / league_avg_ps
        local = PARAMS.BROADCAST_LOCAL_BASE * ps_factor * team.market_size_factor
        return national + local


class CostModel:
    @staticmethod
    def calculate_payroll_cost(team: Team) -> float:
        return team.total_payroll

    @staticmethod
    def calculate_luxury_tax(payroll: float) -> float:
        excess = payroll - PARAMS.SALARY_CAP
        if excess <= 0:
            return 0.0
        return PARAMS.LUXURY_TAX_RATE * excess

    @staticmethod
    def calculate_fixed_costs(year_index: int) -> float:
        base_ops = 50.0
        return base_ops * ((1 + PARAMS.INFLATION_RATE) ** year_index)

    @staticmethod
    def calculate_total_cost(team: Team, year_index: int) -> float:
        payroll = team.total_payroll
        tax = CostModel.calculate_luxury_tax(payroll)
        fixed = CostModel.calculate_fixed_costs(year_index)
        return payroll + tax + fixed

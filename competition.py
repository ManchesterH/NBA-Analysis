import random
from src.team import Team
from src.config import PARAMS
from src.utils import pythagorean_win_prob

class PlayoffSimulator:
    @staticmethod
    def get_boss_team_ps(round_num: int) -> float:
        scaling = 26.6
        if round_num == 1:
            return PARAMS.BOSS_PER_ROUND_1 * scaling
        elif round_num == 2:
            return PARAMS.BOSS_PER_ROUND_2 * scaling
        elif round_num == 3:
            return PARAMS.BOSS_PER_ROUND_3 * scaling
        elif round_num == 4:
            return PARAMS.BOSS_PER_FINALS * scaling
        return 400.0

    @staticmethod
    def simulate_playoff_run(team: Team) -> int:
        qualify_threshold = 350.0
        if team.team_ps_adjusted < qualify_threshold:
            return 0
        rounds_won = 0
        current_round = 1
        max_rounds = 4
        while current_round <= max_rounds:
            boss_ps = PlayoffSimulator.get_boss_team_ps(current_round)
            win_prob = pythagorean_win_prob(team.team_ps_adjusted, boss_ps)
            if random.random() < win_prob:
                rounds_won += 1
                current_round += 1
            else:
                break
        return rounds_won

    @staticmethod
    def calculate_expected_rounds(team: Team) -> float:
        if team.team_ps_adjusted < 350.0:
            return 0.0
        prob_reach = 1.0
        expected_rounds = 0.0
        for r in range(1, 5):
            boss_ps = PlayoffSimulator.get_boss_team_ps(r)
            win_prob = pythagorean_win_prob(team.team_ps_adjusted, boss_ps)
            expected_rounds += prob_reach 
            prob_reach *= win_prob
        return expected_rounds

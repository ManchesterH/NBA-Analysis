from src.player import Player
from src.team import Team
from src.simulation import SimulationEngine
from src.optimization import TicketOptimizer, InjuryMDP
from src.config import PARAMS
from src.dataloader import DataLoader
import numpy as np
import os

def main():
    print("=== Dynamic Valuation Model for Professional Basketball Teams (Data-Driven) ===")
    mcm_path = os.path.join(os.path.dirname(__file__), 'MCM_PER_EWA_data - Sheet1.csv')
    old_path = os.path.join(os.path.dirname(__file__), 'sheet.csv')
    if os.path.exists(mcm_path):
        mcm_loader = DataLoader(mcm_path)
        print(f"Calibration Data: {os.path.basename(mcm_path)}")
        stats = mcm_loader.get_calibration_stats()
    else:
        stats = None
    if os.path.exists(old_path):
        roster_loader = DataLoader(old_path, historical_csv_path=mcm_path if os.path.exists(mcm_path) else None)
        print(f"Roster Data: {os.path.basename(old_path)}")
    else:
        print("Error: No roster data file found.")
        return
    if stats is None:
        stats = roster_loader.get_calibration_stats(2025)
    print("\n--- Data Calibration Stats (from MCM) ---")
    print(f"Average PER: {stats['avg_per']:.2f}")
    print(f"Std PER: {stats['std_per']:.2f}")
    print(f"Top 10 Avg PER: {stats['top_10_avg_per']:.2f}")
    print(f"Top 50 Avg PER: {stats['top_50_avg_per']:.2f}")
    print(f"Max PER: {stats['max_per']:.2f}")
    print(f"Max EWA: {stats['max_ewa']:.2f}")
    PARAMS.BOSS_PER_ROUND_1 = stats['avg_per'] * 1.1
    PARAMS.BOSS_PER_ROUND_2 = stats['top_50_avg_per'] * 0.9
    PARAMS.BOSS_PER_ROUND_3 = stats['top_50_avg_per']
    PARAMS.BOSS_PER_FINALS = stats['top_10_avg_per'] * 0.85
    print(f"-> Adjusted Boss PERs: R1={PARAMS.BOSS_PER_ROUND_1:.1f}, R2={PARAMS.BOSS_PER_ROUND_2:.1f}, R3={PARAMS.BOSS_PER_ROUND_3:.1f}, Finals={PARAMS.BOSS_PER_FINALS:.1f}")
    target_team = 'GS' 
    gsw = Team("Golden State Warriors", market_size_factor=1.5)
    print(f"\nLoading 2025 Roster for {target_team}...")
    roster = roster_loader.get_roster_for_team(target_team, 2025)
    if not roster:
        print("No players found. Exiting.")
        return
    for p in roster:
        gsw.add_player(p)
        print(f"  - {p.name:<20} | PER: {p.per:<5.1f} | EWA: {p.ewa:<4.1f} | Est.Sal: ${p.salary}M")
    print(f"Team Payroll: ${gsw.total_payroll:.2f}M")
    print(f"Team Base PS: {gsw.team_ps_base:.2f}")
    print("\n--- Running Single 5-Year Simulation ---")
    sim = SimulationEngine(gsw, years=5)
    result = sim.run_single_path()
    print(f"{'Year':<5} | {'Revenue':<10} | {'Cost':<10} | {'Profit':<10} | {'Value':<10} | {'Playoff':<5}")
    for i in range(5):
        print(f"{result['years'][i]:<5} | "
              f"${result['revenue'][i]:<9.1f}M | "
              f"${result['cost'][i]:<9.1f}M | "
              f"${result['profit'][i]:<9.1f}M | "
              f"${result['franchise_value'][i]:<9.1f}M | "
              f"R{result['playoff_rounds'][i]}")
    print("\n--- Running Monte Carlo Analysis (100 runs) ---")
    mc_results = sim.run_monte_carlo(100)
    final_values = mc_results["final_values"]
    avg_profits = mc_results["avg_profit"]
    print(f"Mean Final Franchise Value: ${np.mean(final_values):.2f}M")
    print(f"Value Std Dev: ${np.std(final_values):.2f}M")
    print(f"Avg Annual Profit: ${np.mean(avg_profits):.2f}M")
    print("\n--- Ticket Pricing Optimization ---")
    p_star = 200.0
    lambd = 0.02
    optimal_price = TicketOptimizer.optimize_price(PARAMS.STADIUM_CAPACITY, p_star, lambd)
    print(f"Optimal Regular Season Ticket Price: ${optimal_price:.2f}")
    print("\n--- Injury Management MDP ---")
    mdp = InjuryMDP(horizon_games=10)
    mdp.solve()
    fatigue = 5
    action = mdp.get_optimal_action(fatigue, True)
    print(f"Optimal Policy for Healthy Player with Fatigue {fatigue}: {action}")
    fatigue = 9
    action = mdp.get_optimal_action(fatigue, True)
    print(f"Optimal Policy for Healthy Player with Fatigue {fatigue}: {action}")

if __name__ == "__main__":
    main()

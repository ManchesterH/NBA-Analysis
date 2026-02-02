from dataclasses import dataclass

@dataclass
class KeyParams:
    INFLATION_RATE: float = 0.03
    SALARY_CAP: float = 136.0
    LUXURY_TAX_THRESHOLD: float = 136.0
    LUXURY_TAX_RATE: float = 1.5 
    MERCH_BASE: float = 40.0
    MERCH_STAR_INCREMENT_TARGET: float = 35.0
    BROADCAST_NATIONAL: float = 110.0
    BROADCAST_LOCAL_BASE: float = 50.0
    SPONSORSHIP_FIXED: float = 20.0
    BRAND_VALUE_DECAY: float = 0.1
    BRAND_VALUE_PLAYOFF_BOOST: float = 10.0
    STADIUM_CAPACITY: int = 18000
    MAX_TICKET_PRICE: float = 500.0
    TICKET_GROWTH_RATE: float = 0.1
    AVERAGE_TICKET_PRICE_BASE: float = 150.0
    WEIGHT_PER: float = 1.0
    WEIGHT_POPULARITY: float = 15.0
    WEIGHT_EWA: float = 3.0
    WEIGHT_POTENTIAL: float = 0.3
    INJURY_RATE_LAMBDA: float = 0.15
    BOSS_PER_ROUND_1: float = 15.9
    BOSS_PER_ROUND_2: float = 27.4
    BOSS_PER_ROUND_3: float = 30.4
    BOSS_PER_FINALS: float = 27.9
    SIMULATION_YEARS: int = 5
    NUM_SIMULATIONS: int = 1000

PARAMS = KeyParams()

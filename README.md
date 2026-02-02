# NBA Franchise Dynamic Valuation Framework

A comprehensive mathematical modeling system for NBA franchise investment analysis, developed for MCM/ICM 2026.

## 🏀 Overview

This project implements a **Dynamic Valuation Framework (DVF)** that integrates:
- Player performance metrics (PER, EWA, Popularity, Potential)
- Stochastic revenue modeling (4-stream with playoff bonuses)
- "Boss Battle" playoff simulation using Pythagorean expectation
- Markov Decision Process (MDP) for injury management
- Multi-objective Pareto optimization for strategy selection

## 📊 Key Results

- **5-Year Expected Value**: $4.82 billion (95% CI: $4.12B – $5.58B)
- **Sharpe Ratio**: 0.82
- **Value at Risk (95%)**: $3.61B floor
- **Model Validation MAE**: 3.8% against recent franchise sales

## 🔬 Mathematical Models

| Model | Application |
|-------|-------------|
| Geometric Brownian Motion (GBM) | Franchise value trajectory |
| Cox-Ingersoll-Ross (CIR) | Interest rate modeling |
| Black-Scholes | Real options valuation |
| Bellman Equation | Dynamic programming optimization |
| Markov Decision Process | Injury management |
| NSGA-II | Multi-objective Pareto optimization |
| Monte Carlo | Risk quantification (10,000 iterations) |

## 📁 Project Structure

```
Solution/
├── src/                          # Core modules
│   ├── player.py                 # Player Score (PS) model
│   ├── financials.py             # Revenue & cost modeling
│   ├── competition.py            # Boss Battle playoff simulation
│   ├── simulation.py             # Monte Carlo engine
│   ├── optimization.py           # MDP solver
│   ├── advanced_models.py        # GBM, CIR, Real Options, Copula
│   ├── dynamic_optimization.py   # Bellman, NSGA-II, Convex optimization
│   ├── team.py                   # Team aggregation
│   ├── config.py                 # Configuration parameters
│   ├── dataloader.py             # Data loading utilities
│   └── utils.py                  # Helper functions
├── main.py                       # Main entry point
├── generate_figures.py           # Visualization generation
├── generate_advanced_figures.py  # Advanced model visualizations
├── mcm_paper.tex                 # LaTeX paper source
├── figures/                      # Generated visualizations (38 figures)
└── *.csv                         # Data files
```

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/nba-franchise-valuation.git
cd nba-franchise-valuation

# Install dependencies
pip install numpy pandas matplotlib scipy python-docx

# Run main simulation
python main.py

# Generate figures
python generate_figures.py
python generate_advanced_figures.py
```

## 📈 Core Formulas

### Player Score (PS)
```
PS = α·PER + β·Pop(x,y) + γ·EWA + δ·Pot(age)
```

### Geometric Brownian Motion
```
dV(t) = μV(t)dt + σV(t)dW(t)
```

### Bellman Optimality
```
V*(s) = max_a [R(s,a) + γ Σ P(s'|s,a)V*(s')]
```

### Pythagorean Win Probability
```
w = PS_team² / (PS_team² + PS_opponent²)
```

## 📊 Sample Outputs

The model generates 38 visualization figures including:
- Player efficiency distributions
- Revenue breakdown analysis
- Playoff probability progressions
- Stochastic value trajectories
- Pareto optimization frontiers
- Risk metrics dashboards

## 📄 Paper

The full MCM paper is available in `mcm_paper.tex` (LaTeX format).

## 📜 License

MIT License

## 🙏 Acknowledgments

- NBA statistical data sources
- MCM/ICM competition organizers

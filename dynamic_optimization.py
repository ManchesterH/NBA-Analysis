import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm, t as t_dist
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class DynamicProgramming:
    def __init__(self, n_years: int = 5, discount_rate: float = 0.08):
        self.n_years = n_years
        self.discount_rate = discount_rate
        self.states = self._generate_states()
        self.actions = ['hold', 'acquire_star', 'trade_veteran', 'develop_rookie', 'load_manage']
    
    def _generate_states(self) -> List[Tuple]:
        states = []
        for roster_quality in ['low', 'medium', 'high', 'elite']:
            for financial_health in ['poor', 'stable', 'strong']:
                for injury_status in ['healthy', 'minor_injury', 'major_injury']:
                    for playoff_position in ['miss', 'first_round', 'contender', 'favorite']:
                        states.append((roster_quality, financial_health, injury_status, playoff_position))
        return states
    
    def transition_probability(self, state: Tuple, action: str) -> Dict[Tuple, float]:
        roster, finance, injury, playoff = state
        transitions = {}
        
        if action == 'acquire_star':
            if finance == 'strong':
                new_roster = {'low': 'medium', 'medium': 'high', 'high': 'elite', 'elite': 'elite'}[roster]
                new_finance = {'strong': 'stable', 'stable': 'poor', 'poor': 'poor'}[finance]
                transitions[(new_roster, new_finance, injury, playoff)] = 0.7
                transitions[(roster, new_finance, injury, playoff)] = 0.3
            else:
                transitions[state] = 1.0
        
        elif action == 'load_manage':
            new_injury = {'healthy': 'healthy', 'minor_injury': 'healthy', 'major_injury': 'minor_injury'}[injury]
            transitions[(roster, finance, new_injury, playoff)] = 0.8
            transitions[state] = 0.2
        
        elif action == 'develop_rookie':
            if roster in ['low', 'medium']:
                transitions[('medium' if roster == 'low' else 'high', finance, injury, playoff)] = 0.4
                transitions[state] = 0.6
            else:
                transitions[state] = 1.0
        
        else:
            transitions[state] = 1.0
        
        return transitions
    
    def reward(self, state: Tuple, action: str) -> float:
        roster, finance, injury, playoff = state
        
        roster_values = {'low': 50, 'medium': 100, 'high': 200, 'elite': 350}
        finance_multiplier = {'poor': 0.7, 'stable': 1.0, 'strong': 1.3}
        injury_penalty = {'healthy': 0, 'minor_injury': -20, 'major_injury': -80}
        playoff_bonus = {'miss': 0, 'first_round': 30, 'contender': 80, 'favorite': 150}
        
        base_reward = roster_values[roster] * finance_multiplier[finance]
        base_reward += injury_penalty[injury] + playoff_bonus[playoff]
        
        action_costs = {
            'hold': 0,
            'acquire_star': -50,
            'trade_veteran': 10,
            'develop_rookie': -15,
            'load_manage': -5
        }
        
        return base_reward + action_costs[action]
    
    def solve_bellman(self, theta: float = 0.01) -> Tuple[Dict, Dict]:
        V = {s: 0.0 for s in self.states}
        policy = {s: 'hold' for s in self.states}
        gamma = 1 / (1 + self.discount_rate)
        
        for _ in range(1000):
            delta = 0
            for s in self.states:
                v = V[s]
                action_values = []
                
                for a in self.actions:
                    transitions = self.transition_probability(s, a)
                    expected_value = sum(p * (self.reward(s, a) + gamma * V.get(s_next, 0))
                                        for s_next, p in transitions.items())
                    action_values.append((expected_value, a))
                
                best_value, best_action = max(action_values, key=lambda x: x[0])
                V[s] = best_value
                policy[s] = best_action
                delta = max(delta, abs(v - V[s]))
            
            if delta < theta:
                break
        
        return V, policy

class StochasticOptimization:
    def __init__(self, objective: callable, constraints: List[Dict], bounds: List[Tuple]):
        self.objective = objective
        self.constraints = constraints
        self.bounds = bounds
    
    def gradient_descent(self, x0: np.ndarray, learning_rate: float = 0.01, 
                        n_iter: int = 1000, epsilon: float = 1e-8) -> Tuple[np.ndarray, float]:
        x = x0.copy()
        
        for _ in range(n_iter):
            grad = self._numerical_gradient(x, epsilon)
            x = x - learning_rate * grad
            x = np.clip(x, [b[0] for b in self.bounds], [b[1] for b in self.bounds])
        
        return x, self.objective(x)
    
    def _numerical_gradient(self, x: np.ndarray, epsilon: float) -> np.ndarray:
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += epsilon
            x_minus[i] -= epsilon
            grad[i] = (self.objective(x_plus) - self.objective(x_minus)) / (2 * epsilon)
        return grad
    
    def simulated_annealing(self, x0: np.ndarray, T0: float = 1000, 
                           cooling_rate: float = 0.995, n_iter: int = 10000) -> Tuple[np.ndarray, float]:
        x = x0.copy()
        best_x = x.copy()
        best_cost = self.objective(x)
        T = T0
        
        for _ in range(n_iter):
            x_new = x + np.random.randn(len(x)) * T * 0.01
            x_new = np.clip(x_new, [b[0] for b in self.bounds], [b[1] for b in self.bounds])
            
            cost = self.objective(x)
            cost_new = self.objective(x_new)
            
            if cost_new < cost or np.random.random() < np.exp((cost - cost_new) / T):
                x = x_new
                if cost_new < best_cost:
                    best_x = x_new.copy()
                    best_cost = cost_new
            
            T *= cooling_rate
        
        return best_x, best_cost
    
    def particle_swarm(self, n_particles: int = 50, n_iter: int = 100,
                      w: float = 0.7, c1: float = 1.5, c2: float = 1.5) -> Tuple[np.ndarray, float]:
        dim = len(self.bounds)
        
        positions = np.random.uniform(
            [b[0] for b in self.bounds],
            [b[1] for b in self.bounds],
            (n_particles, dim)
        )
        velocities = np.random.randn(n_particles, dim) * 0.1
        
        personal_best_pos = positions.copy()
        personal_best_cost = np.array([self.objective(p) for p in positions])
        
        global_best_idx = np.argmin(personal_best_cost)
        global_best_pos = personal_best_pos[global_best_idx].copy()
        global_best_cost = personal_best_cost[global_best_idx]
        
        for _ in range(n_iter):
            r1, r2 = np.random.rand(2)
            
            velocities = (w * velocities + 
                         c1 * r1 * (personal_best_pos - positions) +
                         c2 * r2 * (global_best_pos - positions))
            
            positions = positions + velocities
            positions = np.clip(positions, [b[0] for b in self.bounds], [b[1] for b in self.bounds])
            
            costs = np.array([self.objective(p) for p in positions])
            
            improved = costs < personal_best_cost
            personal_best_pos[improved] = positions[improved]
            personal_best_cost[improved] = costs[improved]
            
            if np.min(costs) < global_best_cost:
                global_best_idx = np.argmin(costs)
                global_best_pos = positions[global_best_idx].copy()
                global_best_cost = costs[global_best_idx]
        
        return global_best_pos, global_best_cost

class RobustOptimization:
    def __init__(self, nominal_objective: callable, uncertainty_set: Dict):
        self.nominal_objective = nominal_objective
        self.uncertainty_set = uncertainty_set
    
    def worst_case_optimization(self, x0: np.ndarray, bounds: List[Tuple],
                               n_scenarios: int = 100) -> Tuple[np.ndarray, float]:
        def robust_objective(x):
            worst_value = float('-inf')
            
            for _ in range(n_scenarios):
                perturbed_params = {}
                for param, (mean, std) in self.uncertainty_set.items():
                    perturbed_params[param] = mean + np.random.randn() * std
                
                value = self.nominal_objective(x, perturbed_params)
                worst_value = max(worst_value, value)
            
            return worst_value
        
        result = minimize(robust_objective, x0, bounds=bounds, method='L-BFGS-B')
        return result.x, result.fun
    
    def chance_constrained(self, x0: np.ndarray, bounds: List[Tuple],
                          constraint_fn: callable, confidence: float = 0.95,
                          n_samples: int = 1000) -> Tuple[np.ndarray, float]:
        def penalized_objective(x):
            base_cost = self.nominal_objective(x, {})
            
            violations = 0
            for _ in range(n_samples):
                perturbed_params = {}
                for param, (mean, std) in self.uncertainty_set.items():
                    perturbed_params[param] = mean + np.random.randn() * std
                
                if constraint_fn(x, perturbed_params) > 0:
                    violations += 1
            
            violation_rate = violations / n_samples
            penalty = max(0, violation_rate - (1 - confidence)) * 1000
            
            return base_cost + penalty
        
        result = minimize(penalized_objective, x0, bounds=bounds, method='L-BFGS-B')
        return result.x, result.fun

class MultiObjectiveOptimizer:
    def __init__(self, objectives: List[callable], bounds: List[Tuple]):
        self.objectives = objectives
        self.bounds = bounds
    
    def weighted_sum(self, weights: np.ndarray, x0: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        def combined_objective(x):
            values = [obj(x) for obj in self.objectives]
            return sum(w * v for w, v in zip(weights, values))
        
        result = minimize(combined_objective, x0, bounds=self.bounds, method='L-BFGS-B')
        final_values = [obj(result.x) for obj in self.objectives]
        
        return result.x, final_values
    
    def epsilon_constraint(self, primary_idx: int, epsilon_bounds: List[Tuple],
                          x0: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        def constrained_objective(x):
            return self.objectives[primary_idx](x)
        
        constraints = []
        for i, (lower, upper) in enumerate(epsilon_bounds):
            if i != primary_idx:
                if lower is not None:
                    constraints.append({'type': 'ineq', 'fun': lambda x, i=i, l=lower: self.objectives[i](x) - l})
                if upper is not None:
                    constraints.append({'type': 'ineq', 'fun': lambda x, i=i, u=upper: u - self.objectives[i](x)})
        
        result = minimize(constrained_objective, x0, bounds=self.bounds, 
                         constraints=constraints, method='SLSQP')
        final_values = [obj(result.x) for obj in self.objectives]
        
        return result.x, final_values
    
    def nsga2(self, pop_size: int = 100, n_generations: int = 100) -> List[Tuple[np.ndarray, List[float]]]:
        dim = len(self.bounds)
        
        population = np.random.uniform(
            [b[0] for b in self.bounds],
            [b[1] for b in self.bounds],
            (pop_size, dim)
        )
        
        for gen in range(n_generations):
            fitness = np.array([[obj(ind) for obj in self.objectives] for ind in population])
            
            fronts = self._fast_non_dominated_sort(fitness)
            
            crowding = self._crowding_distance(fitness, fronts)
            
            new_population = []
            for front in fronts:
                if len(new_population) + len(front) <= pop_size:
                    new_population.extend([population[i] for i in front])
                else:
                    sorted_front = sorted(front, key=lambda i: crowding[i], reverse=True)
                    remaining = pop_size - len(new_population)
                    new_population.extend([population[i] for i in sorted_front[:remaining]])
                    break
            
            population = np.array(new_population)
            
            offspring = self._crossover_mutation(population)
            population = np.vstack([population, offspring])
        
        fitness = np.array([[obj(ind) for obj in self.objectives] for ind in population])
        fronts = self._fast_non_dominated_sort(fitness)
        pareto_front = [(population[i], list(fitness[i])) for i in fronts[0]]
        
        return pareto_front
    
    def _fast_non_dominated_sort(self, fitness: np.ndarray) -> List[List[int]]:
        n = len(fitness)
        domination_count = np.zeros(n)
        dominated_solutions = [[] for _ in range(n)]
        fronts = [[]]
        
        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(fitness[i], fitness[j]):
                    dominated_solutions[i].append(j)
                    domination_count[j] += 1
                elif self._dominates(fitness[j], fitness[i]):
                    dominated_solutions[j].append(i)
                    domination_count[i] += 1
        
        for i in range(n):
            if domination_count[i] == 0:
                fronts[0].append(i)
        
        current_front = 0
        while fronts[current_front]:
            next_front = []
            for i in fronts[current_front]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front += 1
            if next_front:
                fronts.append(next_front)
        
        return fronts[:-1] if fronts[-1] == [] else fronts
    
    def _dominates(self, a: np.ndarray, b: np.ndarray) -> bool:
        return all(a <= b) and any(a < b)
    
    def _crowding_distance(self, fitness: np.ndarray, fronts: List[List[int]]) -> np.ndarray:
        n = len(fitness)
        crowding = np.zeros(n)
        
        for front in fronts:
            if len(front) <= 2:
                for i in front:
                    crowding[i] = float('inf')
                continue
            
            for obj_idx in range(fitness.shape[1]):
                sorted_front = sorted(front, key=lambda i: fitness[i, obj_idx])
                crowding[sorted_front[0]] = float('inf')
                crowding[sorted_front[-1]] = float('inf')
                
                obj_range = fitness[sorted_front[-1], obj_idx] - fitness[sorted_front[0], obj_idx]
                if obj_range == 0:
                    continue
                
                for i in range(1, len(sorted_front) - 1):
                    crowding[sorted_front[i]] += (
                        (fitness[sorted_front[i + 1], obj_idx] - fitness[sorted_front[i - 1], obj_idx]) / obj_range
                    )
        
        return crowding
    
    def _crossover_mutation(self, population: np.ndarray, 
                           crossover_rate: float = 0.9, mutation_rate: float = 0.1) -> np.ndarray:
        n, dim = population.shape
        offspring = np.zeros((n, dim))
        
        for i in range(0, n, 2):
            p1, p2 = population[np.random.choice(n, 2, replace=False)]
            
            if np.random.random() < crossover_rate:
                alpha = np.random.random(dim)
                c1 = alpha * p1 + (1 - alpha) * p2
                c2 = alpha * p2 + (1 - alpha) * p1
            else:
                c1, c2 = p1.copy(), p2.copy()
            
            for c in [c1, c2]:
                if np.random.random() < mutation_rate:
                    mut_idx = np.random.randint(dim)
                    c[mut_idx] += np.random.randn() * 0.1 * (self.bounds[mut_idx][1] - self.bounds[mut_idx][0])
                    c[mut_idx] = np.clip(c[mut_idx], self.bounds[mut_idx][0], self.bounds[mut_idx][1])
            
            offspring[i] = c1
            if i + 1 < n:
                offspring[i + 1] = c2
        
        return offspring

class ConvexOptimization:
    @staticmethod
    def portfolio_optimization(expected_returns: np.ndarray, cov_matrix: np.ndarray,
                              risk_aversion: float = 1.0) -> np.ndarray:
        n = len(expected_returns)
        
        def objective(w):
            portfolio_return = np.dot(w, expected_returns)
            portfolio_risk = np.dot(w, np.dot(cov_matrix, w))
            return -portfolio_return + risk_aversion * portfolio_risk
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        bounds = [(0, 1) for _ in range(n)]
        
        x0 = np.ones(n) / n
        result = minimize(objective, x0, bounds=bounds, constraints=constraints, method='SLSQP')
        
        return result.x
    
    @staticmethod
    def lasso_regression(X: np.ndarray, y: np.ndarray, lambda_reg: float = 1.0) -> np.ndarray:
        n, p = X.shape
        
        def objective(beta):
            residuals = y - X @ beta
            mse = np.mean(residuals ** 2)
            l1_penalty = lambda_reg * np.sum(np.abs(beta))
            return mse + l1_penalty
        
        x0 = np.zeros(p)
        result = minimize(objective, x0, method='L-BFGS-B')
        
        return result.x
    
    @staticmethod
    def ridge_regression(X: np.ndarray, y: np.ndarray, lambda_reg: float = 1.0) -> np.ndarray:
        n, p = X.shape
        I = np.eye(p)
        beta = np.linalg.inv(X.T @ X + lambda_reg * I) @ X.T @ y
        return beta

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm, poisson
from typing import List, Dict, Tuple, Callable
import warnings
warnings.filterwarnings('ignore')

class ParetoFrontier:
    def __init__(self):
        self.solutions = []
        self.objectives = []
    
    def add_solution(self, solution: Dict, objectives: Tuple[float, float]):
        self.solutions.append(solution)
        self.objectives.append(objectives)
    
    def compute_frontier(self) -> List[Tuple[Dict, Tuple[float, float]]]:
        if not self.objectives:
            return []
        
        pareto_front = []
        for i, (sol, obj) in enumerate(zip(self.solutions, self.objectives)):
            dominated = False
            for j, other_obj in enumerate(self.objectives):
                if i != j:
                    if other_obj[0] >= obj[0] and other_obj[1] >= obj[1]:
                        if other_obj[0] > obj[0] or other_obj[1] > obj[1]:
                            dominated = True
                            break
            if not dominated:
                pareto_front.append((sol, obj))
        
        return sorted(pareto_front, key=lambda x: x[1][0])

class AdvancedMDP:
    def __init__(self, states: List[str], actions: List[str], 
                 transition_fn: Callable, reward_fn: Callable,
                 gamma: float = 0.95):
        self.states = states
        self.actions = actions
        self.transition_fn = transition_fn
        self.reward_fn = reward_fn
        self.gamma = gamma
        self.V = {s: 0.0 for s in states}
        self.policy = {s: actions[0] for s in states}
    
    def value_iteration(self, theta: float = 1e-6, max_iter: int = 1000) -> Dict[str, float]:
        for iteration in range(max_iter):
            delta = 0
            for s in self.states:
                v = self.V[s]
                action_values = []
                for a in self.actions:
                    next_states_probs = self.transition_fn(s, a)
                    value = sum(p * (self.reward_fn(s, a, s_next) + self.gamma * self.V[s_next])
                               for s_next, p in next_states_probs.items())
                    action_values.append((value, a))
                
                best_value, best_action = max(action_values, key=lambda x: x[0])
                self.V[s] = best_value
                self.policy[s] = best_action
                delta = max(delta, abs(v - self.V[s]))
            
            if delta < theta:
                break
        
        return self.V
    
    def policy_iteration(self, max_iter: int = 100) -> Dict[str, str]:
        for _ in range(max_iter):
            self._policy_evaluation()
            
            policy_stable = True
            for s in self.states:
                old_action = self.policy[s]
                action_values = []
                for a in self.actions:
                    next_states_probs = self.transition_fn(s, a)
                    value = sum(p * (self.reward_fn(s, a, s_next) + self.gamma * self.V[s_next])
                               for s_next, p in next_states_probs.items())
                    action_values.append((value, a))
                
                best_value, best_action = max(action_values, key=lambda x: x[0])
                self.policy[s] = best_action
                
                if old_action != best_action:
                    policy_stable = False
            
            if policy_stable:
                break
        
        return self.policy
    
    def _policy_evaluation(self, theta: float = 1e-6):
        while True:
            delta = 0
            for s in self.states:
                v = self.V[s]
                a = self.policy[s]
                next_states_probs = self.transition_fn(s, a)
                self.V[s] = sum(p * (self.reward_fn(s, a, s_next) + self.gamma * self.V[s_next])
                               for s_next, p in next_states_probs.items())
                delta = max(delta, abs(v - self.V[s]))
            if delta < theta:
                break

class StochasticDifferentialEquation:
    def __init__(self, mu: Callable, sigma: Callable, x0: float):
        self.mu = mu
        self.sigma = sigma
        self.x0 = x0
    
    def euler_maruyama(self, T: float, n_steps: int, n_paths: int = 1000) -> np.ndarray:
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.x0
        
        for t in range(n_steps):
            x = paths[:, t]
            dW = np.random.normal(0, sqrt_dt, n_paths)
            paths[:, t + 1] = x + self.mu(x, t * dt) * dt + self.sigma(x, t * dt) * dW
        
        return paths
    
    def milstein(self, T: float, n_steps: int, sigma_prime: Callable, n_paths: int = 1000) -> np.ndarray:
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.x0
        
        for t in range(n_steps):
            x = paths[:, t]
            dW = np.random.normal(0, sqrt_dt, n_paths)
            sig = self.sigma(x, t * dt)
            sig_prime = sigma_prime(x, t * dt)
            paths[:, t + 1] = (x + self.mu(x, t * dt) * dt + sig * dW + 
                              0.5 * sig * sig_prime * (dW**2 - dt))
        
        return paths

class GeometricBrownianMotion:
    def __init__(self, S0: float, mu: float, sigma: float):
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
    
    def simulate(self, T: float, n_steps: int, n_paths: int = 1000) -> np.ndarray:
        dt = T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.S0
        
        for t in range(n_steps):
            Z = np.random.standard_normal(n_paths)
            paths[:, t + 1] = paths[:, t] * np.exp((self.mu - 0.5 * self.sigma**2) * dt + 
                                                    self.sigma * np.sqrt(dt) * Z)
        
        return paths
    
    def expected_value(self, T: float) -> float:
        return self.S0 * np.exp(self.mu * T)
    
    def variance(self, T: float) -> float:
        return self.S0**2 * np.exp(2 * self.mu * T) * (np.exp(self.sigma**2 * T) - 1)

class CoxIngersollRoss:
    def __init__(self, r0: float, kappa: float, theta: float, sigma: float):
        self.r0 = r0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
    
    def simulate(self, T: float, n_steps: int, n_paths: int = 1000) -> np.ndarray:
        dt = T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.r0
        
        for t in range(n_steps):
            r = np.maximum(paths[:, t], 0)
            dW = np.random.normal(0, np.sqrt(dt), n_paths)
            paths[:, t + 1] = r + self.kappa * (self.theta - r) * dt + self.sigma * np.sqrt(r) * dW
            paths[:, t + 1] = np.maximum(paths[:, t + 1], 0)
        
        return paths

class KalmanFilter:
    def __init__(self, A: np.ndarray, B: np.ndarray, H: np.ndarray,
                 Q: np.ndarray, R: np.ndarray, x0: np.ndarray, P0: np.ndarray):
        self.A = A
        self.B = B
        self.H = H
        self.Q = Q
        self.R = R
        self.x = x0.copy()
        self.P = P0.copy()
    
    def predict(self, u: np.ndarray = None):
        if u is None:
            u = np.zeros((self.B.shape[1], 1))
        
        self.x = self.A @ self.x + self.B @ u
        self.P = self.A @ self.P @ self.A.T + self.Q
        
        return self.x.copy()
    
    def update(self, z: np.ndarray):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P
        
        return self.x.copy()

class BayesianEstimator:
    def __init__(self, prior_mean: float, prior_var: float):
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.posterior_mean = prior_mean
        self.posterior_var = prior_var
    
    def update(self, observation: float, obs_var: float):
        precision_prior = 1.0 / self.posterior_var
        precision_obs = 1.0 / obs_var
        
        self.posterior_var = 1.0 / (precision_prior + precision_obs)
        self.posterior_mean = self.posterior_var * (precision_prior * self.posterior_mean + 
                                                     precision_obs * observation)
        
        return self.posterior_mean, self.posterior_var
    
    def batch_update(self, observations: List[float], obs_var: float):
        for obs in observations:
            self.update(obs, obs_var)
        return self.posterior_mean, self.posterior_var

class BootstrapConfidenceInterval:
    def __init__(self, data: np.ndarray, statistic: Callable = np.mean):
        self.data = data
        self.statistic = statistic
    
    def compute(self, n_bootstrap: int = 10000, confidence: float = 0.95) -> Tuple[float, float, float]:
        n = len(self.data)
        bootstrap_stats = np.zeros(n_bootstrap)
        
        for i in range(n_bootstrap):
            sample = np.random.choice(self.data, size=n, replace=True)
            bootstrap_stats[i] = self.statistic(sample)
        
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
        point_estimate = self.statistic(self.data)
        
        return point_estimate, lower, upper

class SobolSensitivity:
    def __init__(self, model: Callable, bounds: List[Tuple[float, float]], n_params: int):
        self.model = model
        self.bounds = bounds
        self.n_params = n_params
    
    def first_order_indices(self, n_samples: int = 1024) -> np.ndarray:
        A = np.random.uniform(size=(n_samples, self.n_params))
        B = np.random.uniform(size=(n_samples, self.n_params))
        
        for i in range(self.n_params):
            A[:, i] = A[:, i] * (self.bounds[i][1] - self.bounds[i][0]) + self.bounds[i][0]
            B[:, i] = B[:, i] * (self.bounds[i][1] - self.bounds[i][0]) + self.bounds[i][0]
        
        f_A = np.array([self.model(a) for a in A])
        f_B = np.array([self.model(b) for b in B])
        
        S_i = np.zeros(self.n_params)
        var_Y = np.var(np.concatenate([f_A, f_B]))
        
        for i in range(self.n_params):
            AB_i = B.copy()
            AB_i[:, i] = A[:, i]
            f_AB_i = np.array([self.model(ab) for ab in AB_i])
            
            S_i[i] = np.mean(f_B * (f_AB_i - f_A)) / var_Y if var_Y > 0 else 0
        
        return np.clip(S_i, 0, 1)

class RealOptionsValuation:
    def __init__(self, S0: float, K: float, r: float, sigma: float, T: float):
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
    
    def black_scholes_call(self) -> float:
        d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        
        call_price = self.S0 * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        return call_price
    
    def black_scholes_put(self) -> float:
        d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        
        put_price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S0 * norm.cdf(-d1)
        return put_price
    
    def binomial_tree(self, n_steps: int = 100, option_type: str = 'call') -> float:
        dt = self.T / n_steps
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(self.r * dt) - d) / (u - d)
        
        prices = np.zeros(n_steps + 1)
        for i in range(n_steps + 1):
            prices[i] = self.S0 * (u ** (n_steps - i)) * (d ** i)
        
        if option_type == 'call':
            values = np.maximum(prices - self.K, 0)
        else:
            values = np.maximum(self.K - prices, 0)
        
        for j in range(n_steps - 1, -1, -1):
            for i in range(j + 1):
                values[i] = np.exp(-self.r * dt) * (p * values[i] + (1 - p) * values[i + 1])
        
        return values[0]

class CopulaModel:
    def __init__(self, marginals: List[Callable], theta: float = 2.0):
        self.marginals = marginals
        self.theta = theta
    
    def gaussian_copula(self, correlation_matrix: np.ndarray, n_samples: int = 1000) -> np.ndarray:
        n_vars = len(self.marginals)
        Z = np.random.multivariate_normal(np.zeros(n_vars), correlation_matrix, n_samples)
        U = norm.cdf(Z)
        
        samples = np.zeros_like(U)
        for i, marginal in enumerate(self.marginals):
            samples[:, i] = marginal(U[:, i])
        
        return samples
    
    def clayton_copula(self, n_samples: int = 1000) -> np.ndarray:
        n_vars = len(self.marginals)
        samples = np.zeros((n_samples, n_vars))
        
        for i in range(n_samples):
            V = np.random.gamma(1 / self.theta, 1)
            E = np.random.exponential(1, n_vars)
            U = (1 + E / V) ** (-1 / self.theta)
            
            for j, marginal in enumerate(self.marginals):
                samples[i, j] = marginal(U[j])
        
        return samples

class ExtremValueTheory:
    def __init__(self, data: np.ndarray):
        self.data = data
    
    def block_maxima(self, block_size: int) -> np.ndarray:
        n_blocks = len(self.data) // block_size
        maxima = np.zeros(n_blocks)
        
        for i in range(n_blocks):
            block = self.data[i * block_size:(i + 1) * block_size]
            maxima[i] = np.max(block)
        
        return maxima
    
    def peaks_over_threshold(self, threshold: float) -> np.ndarray:
        exceedances = self.data[self.data > threshold] - threshold
        return exceedances
    
    def estimate_var(self, confidence: float = 0.95) -> float:
        return np.percentile(self.data, confidence * 100)
    
    def estimate_cvar(self, confidence: float = 0.95) -> float:
        var = self.estimate_var(confidence)
        tail = self.data[self.data >= var]
        return np.mean(tail) if len(tail) > 0 else var

class NashEquilibrium:
    def __init__(self, payoff_matrix_1: np.ndarray, payoff_matrix_2: np.ndarray):
        self.payoff_1 = payoff_matrix_1
        self.payoff_2 = payoff_matrix_2
    
    def find_pure_equilibria(self) -> List[Tuple[int, int]]:
        equilibria = []
        n_rows, n_cols = self.payoff_1.shape
        
        for i in range(n_rows):
            for j in range(n_cols):
                is_best_response_1 = self.payoff_1[i, j] == np.max(self.payoff_1[:, j])
                is_best_response_2 = self.payoff_2[i, j] == np.max(self.payoff_2[i, :])
                
                if is_best_response_1 and is_best_response_2:
                    equilibria.append((i, j))
        
        return equilibria
    
    def find_mixed_equilibrium_2x2(self) -> Tuple[np.ndarray, np.ndarray]:
        a, b = self.payoff_2[0, 0], self.payoff_2[0, 1]
        c, d = self.payoff_2[1, 0], self.payoff_2[1, 1]
        
        denom = (a - b - c + d)
        p = (d - b) / denom if denom != 0 else 0.5
        p = np.clip(p, 0, 1)
        
        e, f = self.payoff_1[0, 0], self.payoff_1[0, 1]
        g, h = self.payoff_1[1, 0], self.payoff_1[1, 1]
        
        denom = (e - f - g + h)
        q = (h - g) / denom if denom != 0 else 0.5
        q = np.clip(q, 0, 1)
        
        return np.array([p, 1 - p]), np.array([q, 1 - q])

def compute_information_ratio(returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
    active_returns = returns - benchmark_returns
    tracking_error = np.std(active_returns)
    return np.mean(active_returns) / tracking_error if tracking_error > 0 else 0

def compute_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0

def compute_sortino_ratio(returns: np.ndarray, target_return: float = 0.0) -> float:
    excess_returns = returns - target_return
    downside_returns = returns[returns < target_return]
    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-6
    return np.mean(excess_returns) / downside_std

def compute_max_drawdown(values: np.ndarray) -> float:
    peak = values[0]
    max_dd = 0
    
    for v in values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd
    
    return max_dd

def compute_calmar_ratio(returns: np.ndarray, values: np.ndarray) -> float:
    max_dd = compute_max_drawdown(values)
    annual_return = np.mean(returns) * 252
    return annual_return / max_dd if max_dd > 0 else 0

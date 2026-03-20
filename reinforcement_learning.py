"""
ML Engine — Reinforcement Learning
Simple DQN-based trading agent with discrete actions.
"""
import numpy as np
import pandas as pd
from collections import deque
import random


class TradingEnvironment:
    """Simple trading environment for RL agent."""
    def __init__(self, prices: np.ndarray, window_size: int = 10):
        self.prices = prices
        self.window_size = window_size
        self.current_step = window_size
        self.position = 0  # 0: no position, 1: long
        self.entry_price = 0
        self.total_reward = 0
        self.trades = []

    def reset(self):
        self.current_step = self.window_size
        self.position = 0
        self.entry_price = 0
        self.total_reward = 0
        self.trades = []
        return self._get_state()

    def _get_state(self):
        window = self.prices[self.current_step - self.window_size:self.current_step]
        # Normalize returns as state
        returns = np.diff(window) / window[:-1]
        state = np.append(returns, [self.position])
        return state

    def step(self, action):
        # action: 0=Hold, 1=Buy, 2=Sell
        current_price = self.prices[self.current_step]
        reward = 0
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            self.entry_price = current_price
        elif action == 2 and self.position == 1:  # Sell
            reward = (current_price - self.entry_price) / self.entry_price
            self.trades.append({
                "entry": self.entry_price,
                "exit": current_price,
                "return": round(reward * 100, 2),
            })
            self.position = 0
            self.entry_price = 0
        elif self.position == 1:  # Hold with position
            reward = (current_price - self.prices[self.current_step - 1]) / self.prices[self.current_step - 1] * 0.1
        self.total_reward += reward
        self.current_step += 1
        done = self.current_step >= len(self.prices) - 1
        return self._get_state(), reward, done


class SimpleQLearningAgent:
    """Simple Q-learning trading agent."""
    def __init__(self, state_size: int, action_size: int = 3, learning_rate: float = 0.1,
                 discount: float = 0.95, epsilon: float = 1.0, epsilon_decay: float = 0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        self.memory = deque(maxlen=2000)
        # Q-table approximation using buckets
        self.q_table = {}

    def _discretize_state(self, state: np.ndarray) -> tuple:
        # Discretize to bins
        binned = np.clip(np.round(state * 10) / 10, -2, 2)
        return tuple(binned)

    def act(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        state_key = self._discretize_state(state)
        if state_key not in self.q_table:
            return random.randint(0, self.action_size - 1)
        return int(np.argmax(self.q_table[state_key]))

    def learn(self, state, action, reward, next_state, done):
        state_key = self._discretize_state(state)
        next_key = self._discretize_state(next_state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_key not in self.q_table:
            self.q_table[next_key] = np.zeros(self.action_size)
        target = reward
        if not done:
            target += self.discount * np.max(self.q_table[next_key])
        self.q_table[state_key][action] += self.lr * (target - self.q_table[state_key][action])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_rl_agent(prices: pd.Series, episodes: int = 100, window_size: int = 10) -> dict:
    """Train RL trading agent."""
    result = {"trained": False, "total_return": 0, "trades": [], "episodes": episodes}
    try:
        price_arr = prices.dropna().values
        if len(price_arr) < window_size + 50:
            result["error"] = "Insufficient data"
            return result
        env = TradingEnvironment(price_arr, window_size)
        state_size = window_size  # returns + position
        agent = SimpleQLearningAgent(state_size)
        best_reward = -np.inf
        for ep in range(episodes):
            state = env.reset()
            episode_reward = 0
            while True:
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                agent.learn(state, action, reward, next_state, done)
                episode_reward += reward
                state = next_state
                if done:
                    break
            if episode_reward > best_reward:
                best_reward = episode_reward
        # Final evaluation
        state = env.reset()
        agent.epsilon = 0  # no exploration
        while True:
            action = agent.act(state)
            state, _, done = env.step(action)
            if done:
                break

        result["trained"] = True
        result["total_return"] = round(env.total_reward * 100, 2)
        result["trades"] = env.trades[-20:]  # Last 20 trades
        result["n_trades"] = len(env.trades)
        result["winning_trades"] = len([t for t in env.trades if t["return"] > 0])
        result["win_rate"] = round(result["winning_trades"] / max(len(env.trades), 1) * 100, 1)
        # Current signal
        if len(price_arr) >= window_size:
            final_state = env._get_state()
            final_action = agent.act(final_state)
            result["current_signal"] = {0: "HOLD", 1: "BUY", 2: "SELL"}[final_action]
    except Exception as e:
        result["error"] = str(e)
    return result

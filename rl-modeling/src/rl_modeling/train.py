import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import yaml
import logging
import matplotlib.pyplot as plt

# Configure logging to output to a file
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
with open('training.log', 'w'):
    pass
file_handler = logging.FileHandler('training.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
log.addHandler(file_handler)

class CryptoTradingEnv(gym.Env):
    """
    A Gymnasium environment for crypto trading that loads its config (columns, etc.)
    from a YAML file.

    Actions:
        0 -> Hold
        1 -> Buy (go "all-in" with available balance, using ASK price)
        2 -> Sell (sell all held crypto, using BID price)

    Observation:
        A window of the last N bars, containing the columns specified in the YAML config.
        Shape = (window_size, number_of_columns).

    Reward:
        The change in net worth between steps.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config_path: str,
        window_size: int = 10,
        initial_balance: float = 10_000.0
    ):
        """
        Args:
            df (DataFrame): Historical price data (must contain columns that appear in config).
            config_path (str): Path to the YAML config file.
            window_size (int): Number of past time steps to include in the observation.
            initial_balance (float): Starting fiat balance.
        """
        super(CryptoTradingEnv, self).__init__()
        
        # -----------------------------
        # Load config from YAML
        # -----------------------------
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        self.columns = config["columns"]        # e.g. ["Open", "High", "Low", "Close"]
        # price_column here is optional if you still want a single reference price.
        # However, we will rely on 'BID' and 'ASK' below for trades.
        self.price_column = config.get("price_column", None)

        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance

        # Action space: 3 discrete actions (Hold, Buy, Sell)
        self.action_space = spaces.Discrete(3)

        # Observation space: last window_size rows, each containing len(self.columns) features
        self.observation_space = spaces.Box(
            low=0,
            high=np.finfo(np.float32).max,
            shape=(window_size, len(self.columns)),
            dtype=np.float32
        )
        log.info(f"Environment initialized with columns: {self.columns}")
        if self.price_column is not None:
            log.info(f"Environment initialized with price column: {self.price_column}")

        # Internal state
        self._reset_internals()

    def _reset_internals(self):
        """Reset internal variables at the start of each episode."""
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.crypto_held = 0.0
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.total_steps = len(self.df) - 1
        self.trade_cost = 0.25
        self.reward_types = []

    def _get_observation(self) -> np.ndarray:
        """
        Returns the last `window_size` rows of the configured columns.
        """
        # Window from [current_step - window_size, current_step-1]
        frames = self.df.loc[self.current_step - self.window_size : self.current_step - 1, self.columns]
        return frames.to_numpy(dtype=np.float32)

    def _get_ask_price(self) -> float:
        """Return the current ask price."""
        return float(self.df.loc[self.current_step, "ASK"])

    def _get_bid_price(self) -> float:
        """Return the current bid price."""
        return float(self.df.loc[self.current_step, "BID"])

    def step(self, action: int):
        """
        Execute one time step within the environment.
        Actions:
            0 = Hold
            1 = Buy (all-in) using current ASK
            2 = Sell (all-out) using current BID
        """
        ask_price = self._get_ask_price()
        bid_price = self._get_bid_price()

        # Assertions
        if ask_price <= 0 or np.isnan(ask_price):
            print(f"WARNING: ask_price invalid at step {self.current_step}: {ask_price}")
        if bid_price <= 0 or np.isnan(bid_price):
            print(f"WARNING: bid_price invalid at step {self.current_step}: {bid_price}")

        # Execute action
        if action == 1:  # Buy (all-in at ASK)
            if self.balance > 0:
                if ask_price > 0:
                    # Buy as much crypto as we can with the current fiat balance
                    self.crypto_held += self.balance / ask_price
                    self.balance = 0.0

        elif action == 2:  # Sell (all-out at BID)
            if self.crypto_held > 0:
                # Convert all crypto to fiat at the bid price
                self.balance += self.crypto_held * bid_price
                self.crypto_held = 0.0

        # 0 = Hold → do nothing

        # Update net worth (value any held crypto at the bid price)
        self.net_worth = self.balance + self.crypto_held * bid_price

        # Reward: net worth change
        reward = self.net_worth - self.prev_net_worth
        log.info(f"Step: {self.current_step}:")
        log.info(f"\t Action: {action}, Ask Price: {ask_price:.2f}, Bid Price: {bid_price:.2f}")
        log.info(f"\t AReward: {reward:.2f}, Net Worth: {self.net_worth:.2f}, Prev Net Worth: {self.prev_net_worth:.2f}")
        self.prev_net_worth = self.net_worth

        # Advance step
        self.current_step += 1

        # Check if done
        terminated = (self.current_step >= self.total_steps)
        truncated = False

        # Next observation
        obs = self._get_observation()

        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_internals()
        return self._get_observation(), {}

    def render(self):
        """Optional: Print relevant info or produce some visualization."""
        print(f"Step: {self.current_step}")
        print(f"Balance: {self.balance:.2f}, Held: {self.crypto_held:.6f}, Net Worth: {self.net_worth:.2f}")


# ============================================================================
# Example usage below
# ============================================================================
if __name__ == "__main__":

    df = pd.read_csv("btc_usd_training.csv", parse_dates=["TIMESTAMP"])
    df.sort_index(inplace=True)

    # Drop any NaNs
    # Check for NaN values and print warnings
    if df.isna().any().any():
        nan_rows = df[df.isna().any(axis=1)]
        log.warning(f"Found NaN values in the following rows:\n{nan_rows}")
        df = df.dropna()
        log.info("Dropped rows with NaN values.")

    # Example DataFrame must have at least 'ASK' and 'BID' columns
    df = df[["ASK","ASK_QTY","BID","BID_QTY","CHANGE","CHANGE_PCT","HIGH","LAST","LOW","SYMBOL","VOLUME"]]
    random_index_range = np.arange(0, len(df), 1)
    print(f"index_range: {random_index_range}")
    df = df.iloc[:100_000]  # Limit to 10,000 rows for training
    # Create Gymnasium environment
    def make_env():
        return CryptoTradingEnv(df=df, window_size=10000, initial_balance=10000.0, config_path="config.yaml")

    # Wrap in a DummyVecEnv for Stable-Baselines3
    vec_env = DummyVecEnv([make_env])

    # Instantiate DQN
    model = DQN(
        policy="MlpPolicy", 
        env=vec_env,
        learning_rate=1e-3,
        buffer_size=10000,
        batch_size=32,
        learning_starts=1000,
        verbose=1
    )

    # Train the agent
    model.learn(total_timesteps=10_000)

    # Test the trained model
    test_env = make_env()
    obs, _ = test_env.reset()
    done = False
    counter = 0
    total_reward = 0
    total_reward_list = []
    while not done:
        counter += 1
        # Choose the best action from the model
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = test_env.step(action)
        total_reward += reward
        total_reward_list.append(reward)
        if counter % 1000 == 0:
            test_env.render()
            print(f"Total reward so far: {total_reward:.2f}")
        done = terminated or truncated
    print(test_env.reward_types)
    log.info(f"Total reward from test episode: {total_reward:.2f}")
    model.save("dqn_crypto_trading")
    
    print(f"Total reward from test episode: {total_reward:.2f}")

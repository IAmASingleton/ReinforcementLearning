import os
import random
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium import Env, spaces
import numpy as np
from fhtw_hex import hex_engine as engine

# Define constants
MODEL_PATH = 'PPO50000.zip'
MODEL_PATH_100000 = 'PPO100000.zip'
MODEL_PATH_150000 = 'PPO150000.zip'
MODEL_PATH_200000 = 'PPO200000.zip'
MODEL_PATH_250000 = 'PPO250000.zip'

MOVES_PER_GAME = 60.5  # max possible moves per game
TOTAL_GAMES = 50000    # max possible games per training session
TOTAL_TIMESTEPS = int(TOTAL_GAMES * MOVES_PER_GAME)  # Adjusted total timesteps

class HexEnv(Env):
    def __init__(self, opponent=None):
        super(HexEnv, self).__init__()
        self.game = engine.hexPosition()
        self.action_space = spaces.Discrete(self.game.size * self.game.size)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.game.size, self.game.size, 1), dtype=int)
        self.current_player = 1  # Player 1 starts
        self.opponent = opponent  # Opponent agent

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game = engine.hexPosition()
        self.current_player = 1
        return np.array(self.game.board).reshape(self.game.size, self.game.size, 1), {}

    def step(self, action):
        coordinates = (action // self.game.size, action % self.game.size)
        if self.game.board[coordinates[0]][coordinates[1]] != 0:
            # Invalid move, penalize the agent and end the episode
            reward = -1
            done = True
            return np.array(self.game.board).reshape(self.game.size, self.game.size, 1), reward, done, False, {}
        self.game.moove(coordinates)
        reward = 1 if self.game.winner == self.current_player else -1 if self.game.winner == -self.current_player else 0
        done = self.game.winner != 0
        self.current_player = -self.current_player  # Switch player

        # Opponent move
        if not done and self.opponent is not None and self.current_player == -1:
            opponent_action = self.opponent(self.game.board, self.game.get_action_space())
            self.game.moove(opponent_action)
            reward = 1 if self.game.winner == self.current_player else -1 if self.game.winner == -self.current_player else 0
            done = self.game.winner != 0
            self.current_player = -self.current_player  # Switch player again

        return np.array(self.game.board).reshape(self.game.size, self.game.size, 1), reward, done, False, {}

    def render(self, mode='human'):
        self.game.print()

def create_ppo_agent(params):
    env = DummyVecEnv([lambda: HexEnv()])
    return PPO('MlpPolicy', env, **params, verbose=0)

def create_or_load_ppo_agent():
    if os.path.exists(MODEL_PATH):
        print("Loading existing model PPO50000...")
        agent = PPO.load(MODEL_PATH)
    else:
        print("Training a new model PPO50000...")
        env = DummyVecEnv([lambda: HexEnv()])
        agent = PPO('MlpPolicy', env, verbose=1, n_steps=2048)
        agent.learn(total_timesteps=TOTAL_TIMESTEPS)
        agent.save(MODEL_PATH)
    return agent

agent_ppo_instance = create_or_load_ppo_agent()

def agent_ppo(board, action_set):
    obs = np.array(board).reshape((1, len(board), len(board), 1))
    valid_action_found = False
    while not valid_action_found:
        action, _ = agent_ppo_instance.predict(obs, deterministic=True)
        row, col = divmod(action[0], len(board))

        # Ensure the selected action is valid
        if board[row][col] == 0:
            valid_action_found = True
        else:
            # If invalid, choose a valid action from the action set
            valid_actions = [act for act in action_set if board[act[0]][act[1]] == 0]
            if valid_actions:
                return valid_actions[0]  # Select the first valid action
            else:
                raise ValueError("No valid actions available!")
    return (row, col)

# Theis is the default Agent given
def agent(board, action_set):
    return action_set[0]


def create_or_load_ppo_agent_100000():
    if os.path.exists(MODEL_PATH_100000):
        print("Loading existing model PPO100000...")
        Agent_PPO100000 = PPO.load(MODEL_PATH_100000)
    else:
        print("Training a new model PPO100000...")
        env = DummyVecEnv([lambda: HexEnv()])
        Agent_PPO5000 = PPO.load(MODEL_PATH)  # Load the existing PPO5000 model
        opponent = PPO.load(MODEL_PATH)  # Load the existing PPO5000 model as opponent
        env = DummyVecEnv([lambda: HexEnv(opponent=lambda board, action_set: agent_ppo(board, action_set))])
        Agent_PPO100000 = PPO('MlpPolicy', env, verbose=1, n_steps=2048)
        Agent_PPO100000.set_parameters(Agent_PPO5000.get_parameters())  # Initialize with PPO5000 parameters
        Agent_PPO100000.learn(total_timesteps=TOTAL_TIMESTEPS)
        Agent_PPO100000.save(MODEL_PATH_100000)
    return Agent_PPO100000

# Instantiate the second agent
agent_ppo_instance2 = create_or_load_ppo_agent_100000()

def agent_ppo2(board, action_set):
    obs = np.array(board).reshape((1, len(board), len(board), 1))
    valid_action_found = False
    while not valid_action_found:
        action, _ = agent_ppo_instance2.predict(obs, deterministic=True)
        row, col = divmod(action[0], len(board))

        # Ensure the selected action is valid
        if board[row][col] == 0:
            valid_action_found = True
        else:
            # If invalid, choose a valid action from the action set
            valid_actions = [act for act in action_set if board[act[0]][act[1]] == 0]
            if valid_actions:
                return valid_actions[0]  # Select the first valid action
            else:
                raise ValueError("No valid actions available!")
    return (row, col)



# Define the random agent ensuring valid moves
def agent_random(board, action_set):
    valid_actions = [act for act in action_set if board[act[0]][act[1]] == 0]
    return valid_actions[np.random.randint(len(valid_actions))]

def create_or_load_ppo_agent_150000():
    if os.path.exists(MODEL_PATH_150000):
        print("Loading existing model PPO150000...")
        Agent_PPO150000 = PPO.load(MODEL_PATH_150000)
    else:
        print("Training a new model PPO150000...")
        env = DummyVecEnv([lambda: HexEnv()])
        Agent_PPO100000 = PPO.load(MODEL_PATH_100000)  # Load the existing PPO100000 model
        env = DummyVecEnv([lambda: HexEnv(opponent=agent_random)])  # Use agent_random as the opponent
        Agent_PPO150000 = PPO('MlpPolicy', env, verbose=1, n_steps=2048)
        Agent_PPO150000.set_parameters(Agent_PPO100000.get_parameters())  # Initialize with PPO100000 parameters
        Agent_PPO150000.learn(total_timesteps=TOTAL_TIMESTEPS)
        Agent_PPO150000.save(MODEL_PATH_150000)
    return Agent_PPO150000

# Instantiate the third agent
agent_ppo_instance3 = create_or_load_ppo_agent_150000()

def agent_ppo3(board, action_set):
    obs = np.array(board).reshape((1, len(board), len(board), 1))
    valid_action_found = False
    while not valid_action_found:
        action, _ = agent_ppo_instance3.predict(obs, deterministic=True)
        row, col = divmod(action[0], len(board))

        # Ensure the selected action is valid
        if board[row][col] == 0:
            valid_action_found = True
        else:
            # If invalid, choose a valid action from the action set
            valid_actions = [act for act in action_set if board[act[0]][act[1]] == 0]
            if valid_actions:
                return valid_actions[0]  # Select the first valid action
            else:
                raise ValueError("No valid actions available!")
    return (row, col)


# Define the opponent agent that randomly chooses between PPO3, PPO2, and random
def agent_ppo2_or_ppo3_or_random(board, action_set):
    r = random.random()
    if r < 0.33:
        return agent_ppo3(board, action_set)
    elif r < 0.66:
        return agent_ppo2(board, action_set)
    else:
        return agent_random(board, action_set)

def create_or_load_ppo_agent_200000():
    if os.path.exists(MODEL_PATH_200000):
        print("Loading existing model PPO200000...")
        Agent_PPO200000 = PPO.load(MODEL_PATH_200000)
    else:
        print("Training a new model PPO200000...")
        env = DummyVecEnv([lambda: HexEnv(opponent=agent_ppo2_or_ppo3_or_random)])
        Agent_PPO150000 = PPO.load(MODEL_PATH_150000)  # Load the existing PPO150000 model
        Agent_PPO200000 = PPO('MlpPolicy', env, verbose=1, n_steps=2048)
        Agent_PPO200000.set_parameters(Agent_PPO150000.get_parameters())  # Initialize with PPO150000 parameters
        Agent_PPO200000.learn(total_timesteps=TOTAL_TIMESTEPS)
        Agent_PPO200000.save(MODEL_PATH_200000)
    return Agent_PPO200000

# Instantiate the fourth agent
agent_ppo_instance4 = create_or_load_ppo_agent_200000()

def agent_ppo4(board, action_set):
    obs = np.array(board).reshape((1, len(board), len(board), 1))
    valid_action_found = False
    while not valid_action_found:
        action, _ = agent_ppo_instance4.predict(obs, deterministic=True)
        row, col = divmod(action[0], len(board))

        # Ensure the selected action is valid
        if board[row][col] == 0:
            valid_action_found = True
        else:
            # If invalid, choose a valid action from the action set
            valid_actions = [act for act in action_set if board[act[0]][act[1]] == 0]
            if valid_actions:
                return valid_actions[0]  # Select the first valid action
            else:
                raise ValueError("No valid actions available!")
    return (row, col)


def agent_ppo3_or_ppo4(board, action_set):
    if random.random() < 0.5:
        return agent_ppo3(board, action_set)
    else:
        return agent_ppo4(board, action_set)

def create_or_load_ppo_agent_250000():
    if os.path.exists(MODEL_PATH_250000):
        print("Loading existing model PPO250000...")
        Agent_PPO250000 = PPO.load(MODEL_PATH_250000)
    else:
        print("Training a new model PPO250000...")
        env = DummyVecEnv([lambda: HexEnv()])
        Agent_PPO200000 = PPO.load(MODEL_PATH_200000)  # Load the existing PPO200000 model
        env = DummyVecEnv([lambda: HexEnv(opponent=agent_ppo3_or_ppo4)])  # Use the new mixed opponent function
        Agent_PPO250000 = PPO('MlpPolicy', env, verbose=1, n_steps=2048)
        Agent_PPO250000.set_parameters(Agent_PPO200000.get_parameters())  # Initialize with PPO200000 parameters
        Agent_PPO250000.learn(total_timesteps=TOTAL_TIMESTEPS)
        Agent_PPO250000.save(MODEL_PATH_250000)
    return Agent_PPO250000

# Instantiate the fifth agent
agent_ppo_instance5 = create_or_load_ppo_agent_250000()

def agent_ppo5(board, action_set):
    obs = np.array(board).reshape((1, len(board), len(board), 1))
    valid_action_found = False
    while not valid_action_found:
        action, _ = agent_ppo_instance5.predict(obs, deterministic=True)
        row, col = divmod(action[0], len(board))

        # Ensure the selected action is valid
        if board[row][col] == 0:
            valid_action_found = True
        else:
            # If invalid, choose a valid action from the action set
            valid_actions = [act for act in action_set if board[act[0]][act[1]] == 0]
            if valid_actions:
                return valid_actions[0]  # Select the first valid action
            else:
                raise ValueError("No valid actions available!")
    return (row, col)



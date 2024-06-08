import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import math
import random
import numpy as np

from gym import Env, spaces
from fhtw_hex import hex_engine as engine
from gym.wrappers import RescaleAction

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

from ppo.facade_PPO import agent_ppo as ppo_agent
from ppo.facade_PPO import agent_ppo2 as ppo_agent_2
from ppo.facade_PPO import agent_ppo5 as ppo_agent_5

MOVES_PER_GAME = 60.5  # max possible moves per game
TOTAL_GAMES = 50000    # max possible games per training session
TOTAL_TIMESTEPS = int(TOTAL_GAMES * MOVES_PER_GAME)  # Adjusted total timesteps

# Theis is the default Agent given
def base_agent(board, action_set):
    return action_set[0]

class HexContinuousEnv(Env):
    def __init__(self, opponent=None):
        super(HexContinuousEnv, self).__init__()
        self.game = engine.hexPosition()
        upper_bound = (self.game.size**2)-0.01
        self.action_space = spaces.Box(low=0, high=upper_bound, shape=(1,1), dtype=float)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.game.size, self.game.size, 1), dtype=int)
        self.current_player = random.choice([1,-1])  # Random player starts
        self.opponent = opponent  # Opponent agent

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game = engine.hexPosition()
        self.current_player = random.choice([1,-1])
        return np.array(self.game.board).reshape(self.game.size, self.game.size, 1), {}

    def step(self, action):
        action_discrete = math.floor(action[0][0])
        coordinates = (action_discrete // self.game.size, action_discrete % self.game.size)
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

        #self.render()
        return np.array(self.game.board).reshape(self.game.size, self.game.size, 1), reward, done, False, {}

    def render(self, mode='human'):
        self.game.print()

    
def create_or_load_td3_agent(games=10000, other_ai=None):
    path = "TD3_{}.zip".format(games)
    
    if os.path.exists(path):
        print("Loading existing model {}...".format(games))
        agent = TD3.load(path)
    else:
        print("Training a new model {}...".format(games))
        env = HexContinuousEnv(opponent=other_ai)
        
        #env = gym.make('Pendulum-v1')
        print(type(env.action_space))
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

        agent = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)
        agent.learn(total_timesteps=games*MOVES_PER_GAME)
        agent.save(path)
    return agent

agent_td3_instance_50000  = create_or_load_td3_agent(50000)

def agent_td3_50000(board, action_set):
    obs = np.array(board).reshape((1, len(board), len(board), 1))
    valid_action_found = False
    while not valid_action_found:
        action, _ = agent_td3_instance_50000.predict(obs, deterministic=True)
        action = math.floor(action[0])
        row, col = divmod(action, len(board))
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
    
# Theis is the default Agent given
def agent_predefined(board, action_set):
    return action_set[0]

# Define the random agent ensuring valid moves
def agent_random(board, action_set):
    valid_actions = [act for act in action_set if board[act[0]][act[1]] == 0]
    return valid_actions[np.random.randint(len(valid_actions))]

def choose_agent(board, action_set):
    agents = [agent_random, agent_predefined, agent_td3_50000]
    current_agent = random.choice(agents)
    return current_agent(board, action_set)
    
agent_td3_instance_100000 = create_or_load_td3_agent(100000, choose_agent)

def agent_td3_100000(board, action_set):
    obs = np.array(board).reshape((1, len(board), len(board), 1))
    valid_action_found = False
    while not valid_action_found:
        action, _ = agent_td3_instance_100000.predict(obs, deterministic=True)
        action = math.floor(action[0])
        row, col = divmod(action, len(board))

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

# def choose_agent_2(board, action_set):
#     agents = [agent_random, agent_predefined, agent_td3_100000, ppo_agent, ppo_agent_2, ppo_agent_5]
#     current_agent = random.choice(agents)
#     return current_agent(board, action_set)

# agent_td3_instance_150000 = create_or_load_td3_agent(150000, choose_agent_2)

# def agent_td3_150000(board, action_set):
#     obs = np.array(board).reshape((1, len(board), len(board), 1))
#     valid_action_found = False
#     while not valid_action_found:
#         action, _ = agent_td3_instance_100000.predict(obs, deterministic=True)
#         action = math.floor(action[0])
#         row, col = divmod(action, len(board))

#         # Ensure the selected action is valid
#         if board[row][col] == 0:
#             valid_action_found = True
#         else:
#             # If invalid, choose a valid action from the action set
#             valid_actions = [act for act in action_set if board[act[0]][act[1]] == 0]
#             if valid_actions:
#                 return valid_actions[0]  # Select the first valid action
#             else:
#                 raise ValueError("No valid actions available!")
#     return (row, col)

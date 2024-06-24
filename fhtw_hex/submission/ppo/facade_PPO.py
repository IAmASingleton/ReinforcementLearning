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

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv


# Define constants
MODEL_PATH = 'PPO50000.zip'
MODEL_PATH_100000 = 'PPO100000.zip'
MODEL_PATH_150000 = 'PPO150000.zip'
MODEL_PATH_200000 = 'PPO200000.zip'
MODEL_PATH_250000 = 'PPO250000.zip'

MOVES_PER_GAME = 60.5  # max possible moves per game
TOTAL_GAMES = 50000    # max possible games per training session
TOTAL_TIMESTEPS = int(TOTAL_GAMES * MOVES_PER_GAME)  # Adjusted total timesteps

# Theis is the default Agent given
def agent_predefined(board, action_set):
    return action_set[0]

# Define the random agent ensuring valid moves
def agent_random(board, action_set):
    valid_actions = [act for act in action_set if board[act[0]][act[1]] == 0]
    if len(valid_actions):
        #print("Number of valid actions: {}".format(len(valid_actions)))
        return valid_actions[np.random.randint(len(valid_actions))]
    else:
        return None
  

class HexEnv(Env):
    def __init__(self, opponent=None):
        super(HexEnv, self).__init__()
        self.game = engine.hexPosition()
        self.action_space = spaces.Discrete(self.game.size * self.game.size)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.game.size, self.game.size, 1), dtype=int)
        
        self.current_player = random.choice([1,-1])  # Random player starts
        self.opponent = opponent  # Opponent agent
        
        if self.current_player == -1:
            print(self.opponent)
            first_action_white = self.opponent(
                self.game.board, 
                self.game.get_action_space(recode_black_as_white=False))
            self.game.moove(first_action_white)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.game = engine.hexPosition()
        self.current_player = random.choice([1,-1])
    
        if self.current_player == -1:
            first_action_white = self.opponent(
                self.game.board, 
                self.game.get_action_space(recode_black_as_white=False))
            self.game.moove(first_action_white)
        
        return np.array(self.game.board).reshape(self.game.size, self.game.size, 1), {}

    
    def step(self, action):
        coordinates_player = self.game.scalar_to_coordinates (action)
        if self.current_player == 1:
            action_player = coordinates_player
            
            possible_actions = self.game.get_action_space(recode_black_as_white=True);
            if len(possible_actions) > 0:
                #print("Number of possible actions: {}".format(len(possible_actions)))
                action_opponent = self.opponent(self.game.recode_black_as_white(self.game.board), possible_actions)
                action_opponent = self.game.recode_coordinates(action_opponent)
            else:
                reward = 0
                done = True
                return np.array(self.game.board).reshape(self.game.size, self.game.size, 1), reward, done, False, {}
        else:
            
            action_player = coordinates_player
            
            possible_actions = self.game.get_action_space(recode_black_as_white=False);
            action_opponent=None
            if len(possible_actions) > 0:
                #print("Number of possible actions: {}".format(len(possible_actions)))
                action_opponent = self.opponent(self.game.board, possible_actions)
            else:
                reward = 0
                done = True
                return np.array(self.game.board).reshape(self.game.size, self.game.size, 1), reward, done, False, {}
                
        if self.game.board[action_player[0]][action_player[1]] != 0:
            # Invalid move, penalize the agent and end the episode
            reward = -1
            done = True
            return np.array(self.game.board).reshape(self.game.size, self.game.size, 1), reward, done, False, {}
        
        self.game.moove(action_player)
        reward = 1 if self.game.winner == self.current_player else -1 if self.game.winner == -self.current_player else 0
        done = self.game.winner != 0
        
        if not done:            
            if self.game.board[action_opponent[0]][action_opponent[1]] != 0:
                # Invalid move, penalize the agent and end the episode
                reward = -1
                done = True
                return np.array(self.game.board).reshape(self.game.size, self.game.size, 1), reward, done, False, {}
           
            self.game.moove(action_opponent)
            reward = 1 if self.game.winner == self.current_player else -1 if self.game.winner == -self.current_player else 0
            done = self.game.winner != 0

        #self.render()
        return np.array(self.game.board).reshape(self.game.size, self.game.size, 1), reward, done, False, {}

    # def step(self, action):
    #     coordinates = (action // self.game.size, action % self.game.size)
    #     if self.game.board[coordinates[0]][coordinates[1]] != 0:
    #         # Invalid move, penalize the agent and end the episode
    #         reward = -1
    #         done = True
    #         return np.array(self.game.board).reshape(self.game.size, self.game.size, 1), reward, done, False, {}
    #     self.game.moove(coordinates)
    #     reward = 1 if self.game.winner == self.current_player else -1 if self.game.winner == -self.current_player else 0
    #     done = self.game.winner != 0
    #     self.current_player = -self.current_player  # Switch player

    #     # Opponent move
    #     if not done and self.opponent is not None and self.current_player == -1:
    #         opponent_action = self.opponent(self.game.board, self.game.get_action_space())
    #         self.game.moove(opponent_action)
    #         reward = 1 if self.game.winner == self.current_player else -1 if self.game.winner == -self.current_player else 0
    #         done = self.game.winner != 0
    #         self.current_player = -self.current_player  # Switch player again

    #     return np.array(self.game.board).reshape(self.game.size, self.game.size, 1), reward, done, False, {}

    def render(self, mode='human'):
        self.game.print()

def choose_agent(board, action_set):
    agents = [agent_random, agent_predefined]
    current_agent = random.choice(agents)
    return current_agent(board, action_set)

def create_or_load_ppo_agent(games=5000, other_ai=choose_agent):
    path = "PPO_{}.zip".format(games)
    
    if os.path.exists(path):
        print("Loading existing model {}...".format(games))
        agent = PPO.load(path)
    else:
        print("Training a new model {}...")
        env = HexEnv(opponent=other_ai)
        agent = PPO('MlpPolicy', env, verbose=1, n_steps=2048)
        agent.learn(total_timesteps=games*MOVES_PER_GAME)
        agent.save(path)
    return agent

# def create_ppo_agent(params):
#     env = DummyVecEnv([lambda: HexEnv()])
#     return PPO('MlpPolicy', env, **params, verbose=0)

# def create_or_load_ppo_agent():
#     if os.path.exists(MODEL_PATH):
#         print("Loading existing model PPO50000...")
#         agent = PPO.load(MODEL_PATH)
#     else:
#         print("Training a new model PPO50000...")
#         env = DummyVecEnv([lambda: HexEnv()])
#         agent = PPO('MlpPolicy', env, verbose=1, n_steps=2048)
#         agent.learn(total_timesteps=TOTAL_TIMESTEPS)
#         agent.save(MODEL_PATH)
#     return agent

agent_ppo_instance = create_or_load_ppo_agent(games=100001, other_ai=choose_agent)

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
            #print("Number of valid actions: {}".format(len(valid_actions)))
            if valid_actions:
                return valid_actions[0]  # Select the first valid action
            else:
                raise ValueError("No valid actions available!")
    return (row, col)

def choose_agent_2(board, action_set):
    agents = [agent_random, agent_predefined, agent_ppo]
    current_agent = random.choice(agents)
    return current_agent(board, action_set)

# Instantiate the second agent
agent_ppo_instance_2 = create_or_load_ppo_agent(games=100002, other_ai=choose_agent_2)

def agent_ppo2(board, action_set):
    obs = np.array(board).reshape((1, len(board), len(board), 1))
    valid_action_found = False
    while not valid_action_found:
        action, _ = agent_ppo_instance_2.predict(obs, deterministic=True)
        row, col = divmod(action[0], len(board))

        # Ensure the selected action is valid
        if board[row][col] == 0:
            valid_action_found = True
        else:
            # If invalid, choose a valid action from the action set
            valid_actions = [act for act in action_set if board[act[0]][act[1]] == 0]
            #print("Number of valid actions: {}".format(len(valid_actions)))
            if valid_actions:
                return valid_actions[0]  # Select the first valid action
            else:
                raise ValueError("No valid actions available!")
    return (row, col)

def choose_agent_3(board, action_set):
    agents = [agent_random, agent_predefined, agent_ppo, agent_ppo2]
    current_agent = random.choice(agents)
    return current_agent(board, action_set)

# Instantiate the second agent
agent_ppo_instance_3 = create_or_load_ppo_agent(games=100003, other_ai=choose_agent_3)

def agent_ppo3(board, action_set):
    obs = np.array(board).reshape((1, len(board), len(board), 1))
    valid_action_found = False
    while not valid_action_found:
        action, _ = agent_ppo_instance_2.predict(obs, deterministic=True)
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

def create_or_load_a2c_agent(games=5000, other_ai=choose_agent):
    path = "A2C_{}.zip".format(games)
    
    if os.path.exists(path):
        print("Loading existing model {}...".format(games))
        agent = A2C.load(path)
    else:
        print("Training a new model {}...")
        env = HexEnv(opponent=other_ai)
        agent = A2C('MlpPolicy', env, verbose=1, n_steps=2048)
        agent.learn(total_timesteps=games*MOVES_PER_GAME)
        agent.save(path)
    return agent

agent_a2c_instance = create_or_load_a2c_agent(games=100001, other_ai=choose_agent_3)

def agent_a2c(board, action_set):
    obs = np.array(board).reshape((1, len(board), len(board), 1))
    valid_action_found = False
    while not valid_action_found:
        action, _ = agent_a2c_instance.predict(obs, deterministic=True)
        row, col = divmod(action[0], len(board))

        # Ensure the selected action is valid
        if board[row][col] == 0:
            valid_action_found = True
        else:
            # If invalid, choose a valid action from the action set
            valid_actions = [act for act in action_set if board[act[0]][act[1]] == 0]
            #print("Number of valid actions: {}".format(len(valid_actions)))
            if valid_actions:
                return valid_actions[0]  # Select the first valid action
            else:
                raise ValueError("No valid actions available!")
    return (row, col) 

def choose_agent_4(board, action_set):
    agents = [agent_random, agent_ppo3, agent_a2c]
    current_agent = random.choice(agents)
    return current_agent(board, action_set)

# Instantiate the second agent
agent_ppo_instance_4 = create_or_load_ppo_agent(games=100004, other_ai=choose_agent_4)

def agent_ppo4(board, action_set):
    obs = np.array(board).reshape((1, len(board), len(board), 1))
    valid_action_found = False
    while not valid_action_found:
        action, _ = agent_ppo_instance_2.predict(obs, deterministic=True)
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

agent_a2c_instance_2 = create_or_load_a2c_agent(games=100002, other_ai=choose_agent_3)

def agent_a2c_2(board, action_set):
    obs = np.array(board).reshape((1, len(board), len(board), 1))
    valid_action_found = False
    while not valid_action_found:
        action, _ = agent_a2c_instance.predict(obs, deterministic=True)
        row, col = divmod(action[0], len(board))

        # Ensure the selected action is valid
        if board[row][col] == 0:
            valid_action_found = True
        else:
            # If invalid, choose a valid action from the action set
            valid_actions = [act for act in action_set if board[act[0]][act[1]] == 0]
            #print("Number of valid actions: {}".format(len(valid_actions)))
            if valid_actions:
                return valid_actions[0]  # Select the first valid action
            else:
                raise ValueError("No valid actions available!")
    return (row, col) 

def choose_agent_5(board, action_set):
    agents = [agent_random, agent_ppo4, agent_a2c_2]
    current_agent = random.choice(agents)
    return current_agent(board, action_set)

# Instantiate the second agent
agent_ppo_instance_5 = create_or_load_ppo_agent(games=100005, other_ai=choose_agent_5)

def agent_ppo_5(board, action_set):
    obs = np.array(board).reshape((1, len(board), len(board), 1))
    valid_action_found = False
    while not valid_action_found:
        action, _ = agent_ppo_instance_2.predict(obs, deterministic=True)
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

# def create_or_load_ppo_agent_150000():
#     if os.path.exists(MODEL_PATH_150000):
#         print("Loading existing model PPO150000...")
#         Agent_PPO150000 = PPO.load(MODEL_PATH_150000)
#     else:
#         print("Training a new model PPO150000...")
#         env = DummyVecEnv([lambda: HexEnv()])
#         Agent_PPO100000 = PPO.load(MODEL_PATH_100000)  # Load the existing PPO100000 model
#         env = DummyVecEnv([lambda: HexEnv(opponent=agent_random)])  # Use agent_random as the opponent
#         Agent_PPO150000 = PPO('MlpPolicy', env, verbose=1, n_steps=2048)
#         Agent_PPO150000.set_parameters(Agent_PPO100000.get_parameters())  # Initialize with PPO100000 parameters
#         Agent_PPO150000.learn(total_timesteps=TOTAL_TIMESTEPS)
#         Agent_PPO150000.save(MODEL_PATH_150000)
#     return Agent_PPO150000

# # Instantiate the third agent
# agent_ppo_instance3 = create_or_load_ppo_agent_150000()

# def agent_ppo3(board, action_set):
#     obs = np.array(board).reshape((1, len(board), len(board), 1))
#     valid_action_found = False
#     while not valid_action_found:
#         action, _ = agent_ppo_instance3.predict(obs, deterministic=True)
#         row, col = divmod(action[0], len(board))

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


# # Define the opponent agent that randomly chooses between PPO3, PPO2, and random
# def agent_ppo2_or_ppo3_or_random(board, action_set):
#     r = random.random()
#     if r < 0.33:
#         return agent_ppo3(board, action_set)
#     elif r < 0.66:
#         return agent_ppo2(board, action_set)
#     else:
#         return agent_random(board, action_set)

# def create_or_load_ppo_agent_200000():
#     if os.path.exists(MODEL_PATH_200000):
#         print("Loading existing model PPO200000...")
#         Agent_PPO200000 = PPO.load(MODEL_PATH_200000)
#     else:
#         print("Training a new model PPO200000...")
#         env = DummyVecEnv([lambda: HexEnv(opponent=agent_ppo2_or_ppo3_or_random)])
#         Agent_PPO150000 = PPO.load(MODEL_PATH_150000)  # Load the existing PPO150000 model
#         Agent_PPO200000 = PPO('MlpPolicy', env, verbose=1, n_steps=2048)
#         Agent_PPO200000.set_parameters(Agent_PPO150000.get_parameters())  # Initialize with PPO150000 parameters
#         Agent_PPO200000.learn(total_timesteps=TOTAL_TIMESTEPS)
#         Agent_PPO200000.save(MODEL_PATH_200000)
#     return Agent_PPO200000

# # Instantiate the fourth agent
# agent_ppo_instance4 = create_or_load_ppo_agent_200000()

# def agent_ppo4(board, action_set):
#     obs = np.array(board).reshape((1, len(board), len(board), 1))
#     valid_action_found = False
#     while not valid_action_found:
#         action, _ = agent_ppo_instance4.predict(obs, deterministic=True)
#         row, col = divmod(action[0], len(board))

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


# def agent_ppo3_or_ppo4(board, action_set):
#     if random.random() < 0.5:
#         return agent_ppo3(board, action_set)
#     else:
#         return agent_ppo4(board, action_set)

# def create_or_load_ppo_agent_250000():
#     if os.path.exists(MODEL_PATH_250000):
#         print("Loading existing model PPO250000...")
#         Agent_PPO250000 = PPO.load(MODEL_PATH_250000)
#     else:
#         print("Training a new model PPO250000...")
#         env = DummyVecEnv([lambda: HexEnv()])
#         Agent_PPO200000 = PPO.load(MODEL_PATH_200000)  # Load the existing PPO200000 model
#         env = DummyVecEnv([lambda: HexEnv(opponent=agent_ppo3_or_ppo4)])  # Use the new mixed opponent function
#         Agent_PPO250000 = PPO('MlpPolicy', env, verbose=1, n_steps=2048)
#         Agent_PPO250000.set_parameters(Agent_PPO200000.get_parameters())  # Initialize with PPO200000 parameters
#         Agent_PPO250000.learn(total_timesteps=TOTAL_TIMESTEPS)
#         Agent_PPO250000.save(MODEL_PATH_250000)
#     return Agent_PPO250000

# # Instantiate the fifth agent
# agent_ppo_instance5 = create_or_load_ppo_agent_250000()

# def agent_ppo5(board, action_set):
#     obs = np.array(board).reshape((1, len(board), len(board), 1))
#     valid_action_found = False
#     while not valid_action_found:
#         action, _ = agent_ppo_instance5.predict(obs, deterministic=True)
#         row, col = divmod(action[0], len(board))

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



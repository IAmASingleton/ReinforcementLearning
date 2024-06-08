import sys

import numpy as np
import matplotlib.pyplot as plt

from fhtw_hex.submission.td3.facade_TD3 import agent_predefined, agent_random
from fhtw_hex.submission.td3.facade_TD3 import agent_td3_50000 as td3_agent_1  
from fhtw_hex.submission.td3.facade_TD3 import agent_td3_100000 as td3_agent_2

from fhtw_hex.submission.ppo.facade_PPO import agent_ppo as ppo_agent
from fhtw_hex.submission.ppo.facade_PPO import agent_ppo2 as ppo_agent_2
from fhtw_hex.submission.ppo.facade_PPO import agent_ppo5 as ppo_agent_5
#from fhtw_hex.submission.facade_PPO import agent, agent_ppo_instance, agent_ppo, agent_ppo_instance2, agent_ppo2, agent_ppo_instance3, agent_ppo3, agent_ppo_instance4, agent_ppo4, agent_ppo_instance5, agent_ppo5

# from fhtw_hex.submission.facade_TD3 import agent_td3_150000 as td3_agent_3
# from fhtw_hex.submission.facade_TD3 import agent_td3_200000 as td3_agent_4
from fhtw_hex import hex_engine as engine

# Initialize a game object
game = engine.hexPosition()
def play_games(agent1=None, agent2=None, num_games=100):
    results = {"Agent1 Wins": 0, "Agent2 Wins": 0}
    for _ in range(num_games):
        game = engine.hexPosition()
        game.machine_vs_machine(machine1=agent1, machine2=agent2)
        if game.winner == 1:
            results["Agent1 Wins"] += 1
        else:
            results["Agent2 Wins"] += 1
    return results 

def main():
 
    # Test the provided agent against a random agent
    game.machine_vs_machine(machine1=td3_agent_1, machine2=agent_random)
    game.machine_vs_machine(machine1=agent_random, machine2=td3_agent_1)
    game.machine_vs_machine(machine1=td3_agent_1, machine2=agent_predefined)
    game.machine_vs_machine(machine1=agent_predefined, machine2=td3_agent_1)

    print("-------------------------------------------")



    # Play games between PPO agent and random agent
    td3_vs_random_results = play_games(td3_agent_1, agent_random, num_games=50)
    random_vs_td_results = play_games(agent_random, td3_agent_1, num_games=50)
    
    # Play games between PPO agent and predefined agent
    td3_vs_predefined_results = play_games(td3_agent_1, agent_predefined, num_games=50)
    predefined_vs_td3_results = play_games(agent_predefined, td3_agent_1, num_games=50)
    
    # Combine results for plotting
    combined_results = {
        "TD3 Agent Wins\nvs Random": td3_vs_random_results["Agent1 Wins"] + random_vs_td_results["Agent2 Wins"],
        "Random Agent Wins\nvs TD3": td3_vs_random_results["Agent2 Wins"] + random_vs_td_results["Agent1 Wins"],
        "TD3 Agent Wins\nvs Base": td3_vs_predefined_results["Agent1 Wins"] + predefined_vs_td3_results["Agent2 Wins"],
        "Base Agent Wins\nvs TD3": td3_vs_predefined_results["Agent2 Wins"] + predefined_vs_td3_results["Agent1 Wins"]
        }

    # Plot the results
    labels = combined_results.keys()
    values = combined_results.values()

    plt.bar(labels, values, color=['blue', 'green', 'red', 'purple'])
    plt.xlabel('Agents')
    plt.ylabel('Number of Wins')
    plt.title('TD3 Agent Performance')
    plt.show()

    print("-------------------------------------------")

def main2():     
    
    game.machine_vs_machine(machine1=td3_agent_2, machine2=None)
    game.machine_vs_machine(machine1=None, machine2=td3_agent_2)
    game.machine_vs_machine(machine1=td3_agent_2, machine2=td3_agent_1)
    game.machine_vs_machine(machine1=td3_agent_1, machine2=td3_agent_2)
    
    print("-------------------------------------------")


    # Play games between TD3 agent and random agent
    td3v2_vs_random_results = play_games(td3_agent_2, agent_random, num_games=50)
    random_vs_td3v2_results = play_games(agent_random, td3_agent_2, num_games=50)
    
    # Play games between TD3 agent and TD3 agent
    td3v2_vs_td3v1_results = play_games(td3_agent_2, td3_agent_1, num_games=50)
    td3v1_vs_td3v2_results = play_games(td3_agent_1, td3_agent_2, num_games=50)
    
    # Play games between TD3 agent and PPO agent
    td3v2_vs_ppov1_results = play_games(td3_agent_2, ppo_agent, num_games=50)
    ppov1_vs_td3v2_results = play_games(ppo_agent, td3_agent_2, num_games=50)

    # Combine results for plotting
    combined_results_v2 = {
        "TD3 Agent Wins vs Random": td3v2_vs_random_results["Agent1 Wins"] + random_vs_td3v2_results["Agent2 Wins"],
        "Random Agent Wins vs TD3": td3v2_vs_random_results["Agent2 Wins"] + random_vs_td3v2_results["Agent1 Wins"],
        "TD3(v2) wins vs TD3(v1)": td3v2_vs_td3v1_results["Agent1 Wins"] + td3v1_vs_td3v2_results["Agent2 Wins"],
        "TD3(v1) wins vs TD3(v2)": td3v2_vs_td3v1_results["Agent2 Wins"] + td3v1_vs_td3v2_results["Agent1 Wins"],
        }

    # Plot the results
    labels_v2 = combined_results_v2.keys()
    values_v2 = combined_results_v2.values()

    plt.bar(labels_v2, values_v2, color=['blue', 'green', 'red', 'purple', 'yellow', 'orange'])
    plt.xlabel('Agents')
    plt.ylabel('Number of Wins')
    plt.title('TD3 Agent Performance')
    plt.show()

    

    print("-------------------------------------------")

def main3():
    
    game.machine_vs_machine(machine1=td3_agent_2, machine2=ppo_agent)
    game.machine_vs_machine(machine1=ppo_agent, machine2=td3_agent_2)
    game.machine_vs_machine(machine1=td3_agent_2, machine2=ppo_agent_2)
    game.machine_vs_machine(machine1=ppo_agent_2, machine2=td3_agent_2)
    game.machine_vs_machine(machine1=td3_agent_2, machine2=ppo_agent_5)
    game.machine_vs_machine(machine1=ppo_agent_5, machine2=td3_agent_2)
    
    print("-------------------------------------------")


    # Play games between TD3 agent and random agent
    td3v2_vs_ppov1_results = play_games(td3_agent_2, ppo_agent, num_games=50)
    ppov1_vs_td3v2_results = play_games(ppo_agent, td3_agent_2, num_games=50)
    
    # Play games between TD3 agent and TD3 agent
    td3v2_vs_ppov2_results = play_games(td3_agent_2, ppo_agent_2, num_games=50)
    ppov2_vs_td3v2_results = play_games(ppo_agent_2, td3_agent_2, num_games=50)
    
    # Play games between TD3 agent and PPO agent
    td3v2_vs_ppov5_results = play_games(td3_agent_2, ppo_agent_5, num_games=50)
    ppov5_vs_td3v2_results = play_games(ppo_agent_5, td3_agent_2, num_games=50)

    # Combine results for plotting
    combined_results_v3 = {
        "TD3(v2) wins\nvs PPO(v1)": td3v2_vs_ppov1_results["Agent1 Wins"] + ppov1_vs_td3v2_results["Agent2 Wins"],
        "PPO(v1) wins\nvs TD3(v2)": td3v2_vs_ppov1_results["Agent2 Wins"] + ppov1_vs_td3v2_results["Agent1 Wins"],
        "TD3(v2) wins\nvs PPO(v2)": td3v2_vs_ppov2_results["Agent1 Wins"] + ppov2_vs_td3v2_results["Agent2 Wins"],
        "PPO(v2) wins\nvs TD3(v2)": td3v2_vs_ppov2_results["Agent2 Wins"] + ppov2_vs_td3v2_results["Agent1 Wins"],
        "TD3(v2) wins\nvs PPO(v5)": td3v2_vs_ppov5_results["Agent1 Wins"] + ppov5_vs_td3v2_results["Agent2 Wins"],
        "PPO(v5) wins\nvs TD3(v2)": td3v2_vs_ppov5_results["Agent2 Wins"] + ppov5_vs_td3v2_results["Agent1 Wins"],
        }

    # Plot the results
    labels_v3 = combined_results_v3.keys()
    values_v3 = combined_results_v3.values()

    plt.bar(labels_v3, values_v3, color=['blue', 'green', 'red', 'purple', 'yellow', 'orange'])
    plt.xlabel('Agents')
    plt.ylabel('Number of Wins')
    plt.title('TD3 Agent Performance')
    plt.show()

    print("-------------------------------------------")

if __name__ == "__main__":
    main()
    main2()
    main3()
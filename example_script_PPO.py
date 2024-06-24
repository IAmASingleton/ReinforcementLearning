import sys

import numpy as np
import matplotlib.pyplot as plt

from fhtw_hex.submission.ppo.facade_PPO import agent_predefined, agent_random
# from fhtw_hex.submission.td3.facade_TD3 import agent_td3_v1 as td3_agent_1  
# from fhtw_hex.submission.td3.facade_TD3 import agent_td3_v2 as td3_agent_2
# from fhtw_hex.submission.td3.facade_TD3 import agent_td3_v3 as td3_agent_3

from fhtw_hex.submission.ppo.facade_PPO import agent_ppo as ppo_agent_1
from fhtw_hex.submission.ppo.facade_PPO import agent_ppo2 as ppo_agent_2
from fhtw_hex.submission.ppo.facade_PPO import agent_ppo3 as ppo_agent_3
from fhtw_hex.submission.ppo.facade_PPO import agent_ppo4 as ppo_agent_4
from fhtw_hex.submission.ppo.facade_PPO import agent_ppo_5 as ppo_agent_5

from fhtw_hex.submission.a2c.facade_A2C import agent_a2c as a2c_agent_1
from fhtw_hex.submission.a2c.facade_A2C import agent_a2c_2 as a2c_agent_2


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
    game.machine_vs_machine(machine1=ppo_agent_1, machine2=agent_random)
    game.machine_vs_machine(machine1=agent_random, machine2=ppo_agent_1)
    game.machine_vs_machine(machine1=ppo_agent_1, machine2=agent_predefined)
    game.machine_vs_machine(machine1=agent_predefined, machine2=ppo_agent_1)

    print("-------------------------------------------")



    # Play games between PPO agent and random agent
    ppo_vs_random_results = play_games(ppo_agent_1, agent_random, num_games=50)
    random_vs_ppo_results = play_games(agent_random, ppo_agent_1, num_games=50)
    
    # Play games between PPO agent and predefined agent
    ppo_vs_predefined_results = play_games(ppo_agent_1, agent_predefined, num_games=50)
    predefined_vs_ppo_results = play_games(agent_predefined, ppo_agent_1, num_games=50)
    
    # Combine results for plotting
    combined_results = {
        "PPO(v1)\nwins vs\nRandom": ppo_vs_random_results["Agent1 Wins"] + random_vs_ppo_results["Agent2 Wins"],
        "Random\nwins vs\nPPO(v1)": ppo_vs_random_results["Agent2 Wins"] + random_vs_ppo_results["Agent1 Wins"],
        "PPO(v1)\nwins vs\nBase": ppo_vs_predefined_results["Agent1 Wins"] + predefined_vs_ppo_results["Agent2 Wins"],
        "Base\nwins vs\nPPO(v1)": ppo_vs_predefined_results["Agent2 Wins"] + predefined_vs_ppo_results["Agent1 Wins"]
        }

    # Plot the results
    labels = combined_results.keys()
    values = combined_results.values()

    plt.bar(labels, values, color=['blue', 'green', 'red', 'purple'])
    plt.xlabel('Agents')
    plt.ylabel('Number of Wins')
    plt.title('PPO Agent Performance')
    plt.show()

    print("-------------------------------------------")
 
    game.machine_vs_machine(machine1=ppo_agent_2, machine2=None)
    game.machine_vs_machine(machine1=None, machine2=ppo_agent_2)
    game.machine_vs_machine(machine1=ppo_agent_2, machine2=ppo_agent_1)
    game.machine_vs_machine(machine1=ppo_agent_1, machine2=ppo_agent_2)
    
    print("-------------------------------------------")


    # Play games between PPO agent and random agent
    ppo_vs_random_results = play_games(ppo_agent_2, agent_random, num_games=50)
    random_vs_ppo_results = play_games(agent_random, ppo_agent_2, num_games=50)
    
    # Play games between PPO agent and predefined agent
    ppo_vs_predefined_results = play_games(ppo_agent_2, agent_predefined, num_games=50)
    predefined_vs_ppo_results = play_games(agent_predefined, ppo_agent_2, num_games=50)
    
    # Play games between PPO agent and previous PPO
    ppo_vs_predecessor_results = play_games(ppo_agent_2, ppo_agent_1, num_games=50)
    predecessor_vs_ppo_results = play_games(ppo_agent_1, ppo_agent_2, num_games=50)
    
    
    # Combine results for plotting
    combined_results_v2 = {
        "PPO(v2)\nwins vs\nRandom": ppo_vs_random_results["Agent1 Wins"] + random_vs_ppo_results["Agent2 Wins"],
        "Random\nwins vs\nPPO(v2)": ppo_vs_random_results["Agent2 Wins"] + random_vs_ppo_results["Agent1 Wins"],
        "PPO(v2)\nwins vs\nBase": ppo_vs_predefined_results["Agent1 Wins"] + predefined_vs_ppo_results["Agent2 Wins"],
        "Base\nwins vs\nPPO(v2)": ppo_vs_predefined_results["Agent2 Wins"] + predefined_vs_ppo_results["Agent1 Wins"],
        "PPO(v2)\nwins vs\nPPO(v1)": ppo_vs_predecessor_results["Agent1 Wins"] + predecessor_vs_ppo_results["Agent2 Wins"],
        "PPO(v1)\nwins vs\nPPO(v1)": ppo_vs_predecessor_results["Agent2 Wins"] + predecessor_vs_ppo_results["Agent1 Wins"]
        }


    # Plot the results
    labels_v2 = combined_results_v2.keys()
    values_v2 = combined_results_v2.values()

    plt.bar(labels_v2, values_v2, color=['blue', 'green', 'red', 'purple', 'yellow', 'orange'])
    plt.xlabel('Agents')
    plt.ylabel('Number of Wins')
    plt.title('PPO Agent Performance')
    plt.show() 

    print("-------------------------------------------")
    
    game.machine_vs_machine(machine1=ppo_agent_3, machine2=None)
    game.machine_vs_machine(machine1=None, machine2=ppo_agent_3)
    game.machine_vs_machine(machine1=ppo_agent_3, machine2=ppo_agent_1)
    game.machine_vs_machine(machine1=ppo_agent_1, machine2=ppo_agent_3)
    game.machine_vs_machine(machine1=ppo_agent_3, machine2=ppo_agent_2)
    game.machine_vs_machine(machine1=ppo_agent_2, machine2=ppo_agent_3)
    
    print("-------------------------------------------")


    # Play games between PPO agent and random agent
    ppo_vs_random_results = play_games(ppo_agent_3, agent_random, num_games=50)
    random_vs_ppo_results = play_games(agent_random, ppo_agent_3, num_games=50)
    
    # Play games between PPO agent and predefined agent
    ppoV3_vs_ppoV1_results = play_games(ppo_agent_3, ppo_agent_1, num_games=50)
    ppoV1_vs_ppoV3_results = play_games(ppo_agent_1, ppo_agent_3, num_games=50)
    
    # Play games between PPO agent and previous PPO
    ppoV3_vs_ppoV2_results = play_games(ppo_agent_3, ppo_agent_2, num_games=50)
    ppoV2_vs_ppoV3_results = play_games(ppo_agent_2, ppo_agent_3, num_games=50)
    
    
    # Combine results for plotting
    combined_results_v3 = {
        "PPO(v3)\nwins vs\nRandom": ppo_vs_random_results["Agent1 Wins"] + random_vs_ppo_results["Agent2 Wins"],
        "Random\nwins vs\nPPO(v3)": ppo_vs_random_results["Agent2 Wins"] + random_vs_ppo_results["Agent1 Wins"],
        "PPO(v3)\nwins vs\nPPO(v1)": ppoV3_vs_ppoV1_results["Agent1 Wins"] + ppoV1_vs_ppoV3_results["Agent2 Wins"],
        "PPO(v1)\nwins vs\nPPO(v3)": ppoV3_vs_ppoV1_results["Agent2 Wins"] + ppoV1_vs_ppoV3_results["Agent1 Wins"],
        "PPO(v3)\nwins vs\nPPO(v2)": ppoV3_vs_ppoV2_results["Agent1 Wins"] + ppoV2_vs_ppoV3_results["Agent2 Wins"],
        "PPO(v2)\nwins vs\nPPO(v3)": ppoV3_vs_ppoV2_results["Agent2 Wins"] + ppoV2_vs_ppoV3_results["Agent1 Wins"]
        }


    # Plot the results
    labels_v3 = combined_results_v3.keys()
    values_v3 = combined_results_v3.values()

    plt.bar(labels_v3, values_v3, color=['blue', 'green', 'red', 'purple', 'yellow', 'orange'])
    plt.xlabel('Agents')
    plt.ylabel('Number of Wins')
    plt.title('PPO Agent Performance')
    plt.show() 

    print("-------------------------------------------")
    
    game.machine_vs_machine(machine1=ppo_agent_4, machine2=None)
    game.machine_vs_machine(machine1=None, machine2=ppo_agent_4)
    game.machine_vs_machine(machine1=ppo_agent_4, machine2=ppo_agent_3)
    game.machine_vs_machine(machine1=ppo_agent_3, machine2=ppo_agent_4)
    game.machine_vs_machine(machine1=ppo_agent_4, machine2=a2c_agent_1)
    game.machine_vs_machine(machine1=a2c_agent_1, machine2=ppo_agent_4)
    
    print("-------------------------------------------")


    # Play games between PPO agent and random agent
    ppo_vs_random_results = play_games(ppo_agent_4, agent_random, num_games=50)
    random_vs_ppo_results = play_games(agent_random, ppo_agent_4, num_games=50)
    
    # Play games between PPO agent and predefined agent
    ppoV4_vs_ppoV3_results = play_games(ppo_agent_4, ppo_agent_3, num_games=50)
    ppoV3_vs_ppoV4_results = play_games(ppo_agent_3, ppo_agent_4, num_games=50)
    
    # Play games between PPO agent and previous PPO
    ppoV4_vs_a2cV1_results = play_games(ppo_agent_4, a2c_agent_1, num_games=50)
    a2cV1_vs_ppoV3_results = play_games(a2c_agent_1, ppo_agent_4, num_games=50)
    
    
    # Combine results for plotting
    combined_results_v4 = {
        "PPO(v4)\nwins vs\nRandom": ppo_vs_random_results["Agent1 Wins"] + random_vs_ppo_results["Agent2 Wins"],
        "Random\nwins vs\nPPO(v4)": ppo_vs_random_results["Agent2 Wins"] + random_vs_ppo_results["Agent1 Wins"],
        "PPO(v4)\nwins vs\nPPO(v3)": ppoV4_vs_ppoV3_results["Agent1 Wins"] + ppoV3_vs_ppoV4_results["Agent2 Wins"],
        "PPO(v3)\nwins vs\nPPO(v4)": ppoV4_vs_ppoV3_results["Agent2 Wins"] + ppoV3_vs_ppoV4_results["Agent1 Wins"],
        "PPO(v4)\nwins vs\nA2C(v1)": ppoV4_vs_a2cV1_results["Agent1 Wins"] + a2cV1_vs_ppoV3_results["Agent2 Wins"],
        "A2C(v1)\nwins vs\nPPO(v4)": ppoV4_vs_a2cV1_results["Agent2 Wins"] + a2cV1_vs_ppoV3_results["Agent1 Wins"]
    }

    # Plot the results
    labels_v4 = combined_results_v4.keys()
    values_v4 = combined_results_v4.values()

    plt.bar(labels_v4, values_v4, color=['blue', 'green', 'red', 'purple', 'yellow', 'orange'])
    plt.xlabel('Agents')
    plt.ylabel('Number of Wins')
    plt.title('PPO Agent Performance')
    plt.show() 
    
    print("-------------------------------------------")


    # Play games between PPO agent and random agent
    ppo_vs_random_results = play_games(ppo_agent_5, agent_random, num_games=50)
    random_vs_ppo_results = play_games(agent_random, ppo_agent_5, num_games=50)
 
    # Play games between PPO agent and predefined agent
    ppo_vs_predecessor_results = play_games(ppo_agent_5, ppo_agent_4, num_games=50)
    predecessor_vs_ppo_results = play_games(ppo_agent_4, ppo_agent_5, num_games=50)
 
    # Play games between PPO agent and previous PPO
    ppo_vs_a2c_results = play_games(ppo_agent_5, a2c_agent_2, num_games=50)
    a2c_vs_ppo_results = play_games(a2c_agent_2, ppo_agent_5, num_games=50)
 
 
    # Combine results for plotting
    combined_results_v5 = {
     "PPO(v5)\nwins vs\nRandom": ppo_vs_random_results["Agent1 Wins"] + random_vs_ppo_results["Agent2 Wins"],
     "Random\nwins vs\nPPO(v5)": ppo_vs_random_results["Agent2 Wins"] + random_vs_ppo_results["Agent1 Wins"],
     "PPO(v5)\nwins vs\nPPO(v4)": ppo_vs_predecessor_results["Agent1 Wins"] + predecessor_vs_ppo_results["Agent2 Wins"],
     "PPO(v4)\nwins vs\nPPO(v5)": ppo_vs_predecessor_results["Agent2 Wins"] + predecessor_vs_ppo_results["Agent1 Wins"],
     "PPO(v5)\nwins vs\nA2C(v2)": ppo_vs_a2c_results["Agent1 Wins"] + a2c_vs_ppo_results["Agent2 Wins"],
     "A2C(v2)\nwins vs\nPPO(v5)": ppo_vs_a2c_results["Agent2 Wins"] + a2c_vs_ppo_results["Agent1 Wins"]
     }

     # Plot the results
    labels_v5 = combined_results_v5.keys()
    values_v5 = combined_results_v5.values()

    plt.bar(labels_v5, values_v5, color=['blue', 'green', 'red', 'purple', 'yellow', 'orange'])
    plt.xlabel('Agents')
    plt.ylabel('Number of Wins')
    plt.title('PPO Agent Performance')
    plt.show() 

    print("-------------------------------------------")

if __name__ == "__main__":
    main()
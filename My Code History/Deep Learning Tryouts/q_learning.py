import gym
import random
import numpy as np
import time
#from IPython.display import clear_output


env = gym.make('FrozenLake-v0')

action_space_size = env.action_space.n
state_space_size = env.observation_space.n

q_table = np.zeros((state_space_size,action_space_size))
print(q_table)


num_episodes = 10000
max_steps_per_episodes = 100

learning_rate = 0.1

discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001


rewards_all_episodes = []

for episode in range(num_episodes):
    state = env.reset()

    done = False

    rewards_current_episode = 0

    for step in range(max_steps_per_episodes):
        #exploration-exploitation trade-off
        exploration_rate_threshold = random.uniform(0,1)
        if exploration_rate_threshold > exploration_rate:
            
            action = np.argmax(q_table[state,:])
        else:
            action = env.action_space.sample()

        new_state, reward, done, info = env.step(action)
        

        #Update Q-table for Q(s,a)

        q_table[state,action] = q_table[state, action]*(1 - learning_rate)+ \
        learning_rate * (reward + discount_rate * np.max(q_table[new_state,:]))
        

        state = new_state
        rewards_current_episode += reward

        if done ==True:
            break

        exploration_rate = min_exploration_rate + \
                           (max_exploration_rate - min_exploration_rate)*np.exp( -exploration_decay_rate*episode)
        rewards_all_episodes.append(rewards_current_episode)


#calculte and print the average reward per thoudsand episodes
#rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes),num_episodes/1000)

count = 1000
print("******************Averagereward per thousand episodes **********************")

'''for r in rewards_per_thousand_episodes:
    print(count,":",str(sum(r/1000)))
    count =+ 1000'''

print("******************q-table*****************")
print(q_table)

                        
                            
        
                

                                

        
        
    

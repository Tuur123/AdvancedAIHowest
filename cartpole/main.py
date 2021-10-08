import gym
import numpy as np
import pickle

temperature = 100000 
cooling_rate = 0.993

spread = 0.01
min_spread = 0.0001
max_spread = 0.1

episodes = 10000

previous_weights = np.array([1, 1, 1, 1])

env = gym.make('CartPole-v0')

# with open('weights.pkl', 'rb') as file:
      
#     # Call load method to deserialze
#     previous_weights = pickle.load(file)
  
#     print(f"Using weigts: {previous_weights}")

def computeAction(state, weights):
    p = np.matmul(weights, state)
    if p < 0:
        action = 0
    else:
        action = 1
    return action

for episode in range(episodes):
    
    state = env.reset()
    
    current_reward = 0
    previous_reward = 0


    done = False
    current_weights = previous_weights + np.random.normal(loc=0, scale = spread)

    while not done:
        # env.render()
        action = computeAction(state, current_weights)
        state, reward, done, info = env.step(action)
        current_reward = current_reward + reward

    if current_reward >= previous_reward:
        previous_reward = current_reward
        previous_weights = current_weights
        spread = max(spread/2, min_spread)

    else:
        reward_difference = (current_reward - previous_reward)
        p = np.exp(reward_difference / temperature)

        if  np.random.rand() < p:
            previous_reward = current_reward
            previous_weights = current_weights
            spread = min(spread * 2, max_spread)

    temperature = cooling_rate * temperature

env.close()
print(current_reward)

with open('weights.pkl', 'wb') as file:
      
    # A new file will be created
    pickle.dump(current_weights, file)
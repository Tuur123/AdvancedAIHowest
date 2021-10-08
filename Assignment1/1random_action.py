import gym
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

env = gym.make('CartPole-v0')

steps = []

for episode in range(1000):
    observation = env.reset()
    for t in range(200):
      
        action = env.action_space.sample()        
        observation, reward, done, info = env.step(action)

        if done:
            steps.append(t+1)
            break

env.close()

sns.distplot(steps)
plt.show()

print(f"Average steps per episode: {np.average(steps)}")
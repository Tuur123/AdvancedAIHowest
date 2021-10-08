import gym
import matplotlib.pyplot as plt
from scipy.stats.stats import obrientransform
import seaborn as sns
import numpy as np

env = gym.make('CartPole-v0')

steps = []

for episode in range(1000):
    observation = env.reset()
    for t in range(200):
      
        angle = observation[2] 
        observation, reward, done, info = env.step(0 if angle < 0 else 1)

        if done:
            steps.append(t+1)
            break

env.close()

sns.distplot(steps)
plt.show()

print(f"Average steps per episode: {np.average(steps)}")
import gym
import matplotlib.pyplot as plt
from numpy.lib.function_base import average
from scipy.stats.stats import obrientransform
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

env = gym.make('CartPole-v0')

steps = []
average_reward = []
weight_history = []

def computeAction(state, weights):
    p = np.matmul(weights, state)
    if p < 0:
        action = 0
    else:       
        action = 1
    return action

for iteration in range(1000):

    weights = 2 * np.random.rand(4) -1

    total_reward = 0

    for episode in range(20):
        observation = env.reset()
        for t in range(200):
        
            observation, reward, done, info = env.step(computeAction(observation, weights))
            total_reward += reward
            if done:
                break

    average_reward.append(total_reward / 20)
    weight_history.append([weights, total_reward / 20])

best_weights = weight_history[average_reward.index(np.max(average_reward))]
print(f"Best reward: {np.max(average_reward)} with weights: {best_weights[0]}")

total_reward = 0
for episode in range(1000):
    observation = env.reset()
    for t in range(200):
     
        observation, reward, done, info = env.step(computeAction(observation, best_weights[0]))
        total_reward += reward
        if done:
            break

env.close()

print(f"Average reward over 1000 episodes with best weights: {total_reward / 1000}")
print(f"The most important observation variables will have weights with higher absolute values.")


x_pos = [x[0][1] for x in weight_history if x[1] > 100]
y_pos = [x[0][2] for x in weight_history if x[1] > 100]
z_pos = [x[0][3] for x in weight_history if x[1] > 100]

x_neg = [x[0][1] for x in weight_history if x[1] <= 100]
y_neg = [x[0][2] for x in weight_history if x[1] <= 100]
z_neg = [x[0][3] for x in weight_history if x[1] <= 100]


fig = plt.figure(figsize=(6,6))
ax = Axes3D(fig) 

ax.scatter(x_pos, y_pos, z_pos, c='red', marker='o')
ax.scatter(x_neg, y_neg, z_neg, c='black', marker='^')
ax.set_xlabel('weight 1')
ax.set_ylabel('weight 2')
ax.set_zlabel('weight 3')

plt.show()
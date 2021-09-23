import gym
from time import sleep

env = gym.make('LunarLander-v2')
env.reset()

actions = range(4)

done = False
while done != True:

    env.render()
    obv, reward, done, info = env.step(actions[3])

    stepInfo = {"observation": obv, "reward": reward}
    print(f"{stepInfo}")
    
env.close()
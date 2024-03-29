"""
Reinforcement learning maze example.
Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].
This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.
"""

from maze_env import Maze
from RL_brain import QLearningTable
import pandas as pd

def update():
    result = RL.q_table
    total_reward = 0
    for episode in range(100):
        # initial observation
        # 就是agent的坐标，reset()只返回一个参数
        # 拿到重新开始的初始坐标
        observation = env.reset()
        episode_reward = 0
        step_counter = 0

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(str(observation))

            # RL take action and get next observation and reward
            # step()返回三个参数
            observation_, reward, done = env.step(action)

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_))

            # swap observation
            observation = observation_

            episode_reward = reward
            step_counter +=1 

            # break while loop when end of this episode
            if done:
                break

        total_reward += episode_reward
        result = result.append(pd.Series(
                    [0]*len(RL.actions),
                    index=RL.q_table.columns,
                    name="******",
                ))
        result = result.append(RL.q_table)
        f = open("episode_reward_pair","a")

        episode_reward_pair = '{} use {}steps and get {}, average reward = {}\n'.format(episode, step_counter,episode_reward,total_reward/(episode+1))
        f.write(episode_reward_pair)
        f.close

    result.to_csv('Q.csv', sep=',', header=True, index=True)
    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()
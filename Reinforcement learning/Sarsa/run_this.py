"""
Sarsa is a online updating method for Reinforcement learning.

Unlike Q learning which is a offline updating method, Sarsa is updating while in the current trajectory.

You will see the sarsa is more coward when punishment is close because it cares about all behaviours,
while q learning is more brave because it only cares about maximum behaviour.
"""

from maze_env import Maze
from RL_brain import SarsaTable
import pandas as pd



def update():
    total_reward = 0
    result = RL.q_table
    for episode in range(100):
        # initial observation
        observation = env.reset()

        # RL choose action based on observation
        action = RL.choose_action(str(observation))
        episode_reward = 0
        step_counter = 0

        while True:
            # fresh env
            env.render()

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL choose action based on next observation
            action_ = RL.choose_action(str(observation_))

            # RL learn from this transition (s, a, r, s, a) ==> Sarsa
            RL.learn(str(observation), action, reward, str(observation_), action_)

            # swap observation and action
            observation = observation_
            action = action_
            step_counter +=1
            episode_reward = reward

            # break while loop when end of this episode
            if done:
                break
        result = result.append(pd.Series(
                    [0]*len(RL.actions),
                    index=RL.q_table.columns,
                    name="******",
                ))
        result = result.append(RL.q_table)
        total_reward += episode_reward
        f = open("Sarsa_episode_reward_pair","a")

        episode_reward_pair = '{} use {}steps and get {}, average reward = {}\n'.format(episode, step_counter,episode_reward,total_reward/(episode+1))
        f.write(episode_reward_pair)
        f.close
        

    result.to_csv('a.csv', sep=',', header=True, index=True)
    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = SarsaTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()
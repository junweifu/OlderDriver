# -*- coding: utf-8 -*-

"""
Reinforcement learning maze example.
Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].
This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

from maze_env import Maze
from RL_brain import DeepQNetwork


def update():
    step = 0
    for episode in range(300):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()            

            # RL choose action based on observation
            action = RL.choose_action(str(observation))
            
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL learn from this transition
            RL.store_transiton(observation, action, reward, observation_)

            if (step>200) and (step%5==0):
                RL.learn()
                
            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step +=1

    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = DeepQNetwork(env.n_actions,env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      output_graph=True
                      )

    env.after(100, update)
    env.mainloop()
    RL.plot_cost()
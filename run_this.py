from grid_env_1 import Env
# from grid_env_2 import BuildMaze
from RL_brain import DeepQNetwork
import time

counter = 0


def run_maze():
    # time.sleep(10)
    global counter
    step = 0
    for episode in range(5000):

        # initial observation
        if counter >= 50:
            observation = env.reset()
            while True:
                # observation = env.reset()
                # fresh env
                env.render()

                # RL choose action based on observation
                action = RL.choose_action(observation)

                # RL take action and get next observation and reward
                observation_, reward, done = env.step(action)

                RL.store_transition(observation, action, reward, observation_)

                if (step > 10000) and (step % 5 == 0):  # 记忆量设置为2000
                    RL.learn()

                # swap observation
                observation = observation_

                # break while loop when end of this episode
                if done:
                    # print(step)
                    break
                step += 1
                # print(step)
        else:
            counter += 1
            env.reset()

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # maze game
    # Env._build_maze()
    # Env.on_click()
    # Env.draw_shape()
    # maze = BuildMaze()
    # time.sleep(2)
    env = Env()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.1,
                      reward_decay=0.9,
                      e_greedy=0.8,
                      replace_target_iter=1000,  # 2000
                      memory_size=10000,  # 50000
                      output_graph=False
                      )
    env.after(100, run_maze)  # after()实现简单的定时器功能，after(100,run_maze)表示每隔0.1S执行一次run_maze函数
    env.mainloop()
    # RL.plot_cost()

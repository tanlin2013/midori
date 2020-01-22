import numpy as np
import os, sys
sys.path.insert(0, os.path.abspath('../'))
from lib.DQN import DQN
from envs.recaptcha import Recaptcha


class GuiAgent(Recaptcha):

    def __init__(self):
        super(GuiAgent, self).__init__()
        self.lattice_size = 100
        self.n_actions = 8
        self.n_states = 2
        self.action_space_shape = 0
        self.DQN_params = {
                "batch_size": 32,
                "learning_rate": 0.01,
                "epsilon": 0.9,  # greedy ratio
                "reward_discount": 0.9,
                "target_update_freq": 100,
                "mem_capacity": 2000}
        self.n_episode = 400
    
    def render(self, mode='human'):
        if self.agent is None:
            exp_reward = 0
            self.agent = DQN(self.n_states, self.n_actions, self.action_space_shape, self.DQN_params)
            print('\nCollecting experience...')
            for i in range(self.n_episode):
                self.reset()
                while True:
                    if i < 100:
                        action_idx = self.action_superviser()
                    else:
                        action_idx = self.agent.choose_action(self.state)
                    action = 10*self.action_pool(action_idx)
                    state_i = self.state
                    state_f, done, reward = self.step(action)
                
                    self.agent.store_transition(state_i, action_idx, reward, state_f)

                    exp_reward += reward
                    if self.agent.memory_counter > self.DQN_params["mem_capacity"]:
                        self.agent.learn()
                        if done:
                            print('In epoch: ', i, '| exp_reward: ', round(exp_reward, 2))
 
                    if done:
                        break
        return


if __name__ == '__main__':
    
    agent = GuiAgent()
    print("goal at position:", agent.goal)
    agent.render()

import pyautogui
import numpy as np
import os, sys
sys.path.insert(0, os.path.abspath('../lib'))
from opencv_wrapper import OpencvWrapper
import gym
from gym import logger, spaces

_image_path = os.path.abspath('../envs/test.png')


class Recaptcha:
    """  
    Args:
        max_step_width(sigma): 
            
        max_step_t(tau): 
        
    Attributes:
        observation_space:
            S: 2-dimensional discrete space same as screen resolution
        
        action_space:
            A: U_{4} * W_{sigma} * T_{tau}
            where U_{4} is a discrete space for unit step {left, right, down, up}
            W_{sigma} is a natural number space for step width, and sigma is the maximum width of a step
            T_{tau} is a natural number space for required time of a step, and tau is the maximum time of a step
        
    """
    def __init__(self, max_step_width=30, max_step_t=5):
        self.resolution = pyautogui.size()
        domain = [np.array([0, 1, 1]),
                  np.array([3, max_step_width, max_step_t])]
        
        self.observation_space = spaces.Box(np.zeros(2), np.add(np.array(self.resolution), -1), dtype=np.int)
        self.action_space = spaces.Box(domain[0], domain[1], dtype=np.int)
        self.action_pool = {0: np.array([-1, 0]),
                            1: np.array([1, 0]),
                            2: np.array([0, -1]),
                            3: np.array([0, 1])}
        
        self.state = self.observation_space.sample()
        self.update_state()
        self.steps_beyond_done = None
#        self.alpha = 0.01  # reward euclidean distance decay rate
        self.agent = None
        goal_locs, res = OpencvWrapper.search_image(pyautogui.screenshot(),
                                                    _image_path,
                                                    precision=0.9)
        self.goal = goal_locs[0]
        if len(self.goal) == 0: raise TypeError("Recaptcha icon is not found.")
    
    def update_state(self, t=0):
        pyautogui.moveTo(self.state[0], self.state[1], t)
        return

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        idx, step_width, t = action
        self.state = np.add(self.state, step_width*self.action_pool[idx])
        self.update_state(t)
        done = np.allclose(self.state, self.goal, atol=5, rtol=1e-2)
        done = bool(done)
        
        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0
        
#        eu_dist = np.linalg.norm(self.goal-self.state)
#        reward += np.exp(-self.alpha*eu_dist)
        
        return self.state, done, reward
        
    def reset(self):
        self.state = self.observation_space.sample()
        self.update_state()
        self.steps_beyond_done = None
        return


# =============================================================================
# if __name__ == '__main__':
#      
#     agent = Recaptcha()
#     s = agent.action_space.sample()
#     o = agent.state
#     print(s, o)
#     agent.step(s)
#     print(agent.state)
# =============================================================================
    
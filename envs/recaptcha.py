import pyautogui
import numpy as np
import os, sys
sys.path.insert(0, os.path.abspath('../lib'))
from opencv_wrapper import OpencvWrapper
import gym
from gym import logger
from gym.spaces.space import Space

_image_path = os.path.abspath('../envs/test.png')

class Space2D(Space):
    
    def __init__(self, domain, size=100):
        """2D grid space.
        
        Args:
            domain (list[array_like, array_like]): Allowed value domain of 2D space. 
            size (int): Size of 2D grid space, shape=(size, size).
            
        Attributes:
            x_domain (array_like):
            y_domain (array_like):
            lattice_spacing (np.ndarray):
                
        """
        super(Space2D, self).__init__((size, size), np.float)
        self.x_domain = np.linspace(domain[0][0], domain[0][1], size)
        self.y_domain = np.linspace(domain[1][0], domain[1][1], size)
        self.size = size
        self.lattice_spacing = np.array([np.diff(domain[0])/(size-1), np.diff(domain[1])/(size-1)]).flatten()
    
    def state2coordinate(self, state):
        """
        Define:
            coordinate: (x, y)
            state: 2D zero matrix except the point (x, y) been labeled as 1
            
        """
        idx = np.concatenate((np.where(state==1)), axis=0)
        coordinate = np.array([self.x_domain[idx[0]], self.y_domain[idx[1]]])
        return coordinate
        
    def coordinate2state(self, coordinate):
        state = np.zeros((self.size, self.size))
        idx = (np.where(self.x_domain==coordinate[0]), np.where(self.y_domain==coordinate[1]))
        state[idx] = 1
        return state
        
    def sample(self, seed=None):
        x = np.random.choice(self.x_domain, 1)
        y = np.random.choice(self.y_domain, 1)
        coordinate = np.concatenate((x, y), axis=0)
        return coordinate
        
    def contains(self, coordinate):
        """
        Args: 
            coordinate (np.array, shape=(2, )):
                
        """
        if coordinate[0] in self.x_domain and coordinate[1] in self.y_domain:
            return True
        else:
            return False

class Recaptcha:
    """    
    Attribute:
        state:
            coordinate
            
        observation_space:
            Type: Space
            Axis   Observation               Min    Max
            0      x coordinate of mouse     0      Resolution(x)
            1      y coordinate of mouse     0      Resolution(y)
        
        action_space:
            Type: Space2D
            Axis   Action                   Min    Max
            0      Unit step on x axis      -1     1
            1      Unit step on y axis      -1     1
        
    """
    def __init__(self):
        self.resolution = pyautogui.size()
        self.lattice_size = 100
        self.observation_space = Space2D([[0, self.resolution[0]],
                                          [0, self.resolution[1]]], self.lattice_size)
        self.action_space = Space2D([[-1, 1],
                                     [-1, 1]], 3)
        self.state = self.observation_space.sample()
        self.update_state()
        self.steps_beyond_done = None
        self.alpha = 0.01 # reward euclidean distance decay rate
        self.agent = None
        goal_locs, res = OpencvWrapper.search_image(pyautogui.screenshot(),
                                                    _image_path,
                                                    precision=0.9)
        self.goal = goal_locs[0]
        if len(self.goal) == 0: raise TypeError("Recaptcha icon is not found.")
    
    def update_state(self):
        pyautogui.moveTo(tuple(self.state))
    
    def action_pool(self, idx):
        if isinstance(idx, list): idx = idx[0]
        pool = {0: np.array([-1, 0]),
                1: np.array([1, 0]),
                2: np.array([0, -1]),
                3: np.array([0, 1]),
                4: np.array([1, -1]),
                5: np.array([1, 1]),
                6: np.array([-1, -1]),
                7: np.array([-1, 1])
                }
        return pool[idx]
    
    def action_superviser(self):
        d = (self.state - self.goal)
        if d[0] < 0 and d[1] < 0:
            action_idx = 5
        elif d[0] < 0 and d[1] > 0:
            action_idx = 4
        elif d[0] > 0 and d[1] < 0:
            action_idx = 7
        elif d[0] > 0 and d[1] > 0:
            action_idx = 6
        if d[0] < 0:
            action_idx = 1
        elif d[0] < 1:
            action_idx = 0
        elif d[1] < 0:
            action_idx = 3
        elif d[1] < 1:
            action_idx = 2
        return action_idx
    
    def step(self, action):
        #assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        self.state = np.add(self.state, action)
        self.update_state()
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
        
        eu_dist = np.linalg.norm(self.goal-self.state)
        reward += np.exp(-self.alpha*eu_dist)
        
        return self.state, done, reward
        
    def reset(self):
        self.state = self.observation_space.sample()
        self.update_state()
        self.steps_beyond_done = None
        
# =============================================================================
# if __name__ == '__main__':
#     
#     agent = Recaptcha()
#     s = agent.action_space.sample()
#     print(s)    
# =============================================================================

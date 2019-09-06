import gym
from gym import error, spaces, utils
from gym.utils import seeding
import sys
sys.path.append('/Users/taolin/projects/recatpcha/lib') ## TODO: tmp add
from opencv_wrapper import OpencvWrapper

class Recaptcha(gym.Env):
    
    def __init__(self):
    
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box()
        
        return
        
    def step(self, action):
        
        return
        
        
    def reset(self):
        return
        
    def render(self, mode='human'):
        return
        
    def close(self):
        return
        
if __name__ == '__main__':
    
    a = Recaptcha()
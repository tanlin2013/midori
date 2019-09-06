import pyautogui
import tensorflow as tf
from lib.train_model import QMDP_RCNN
from envs.opencv_wrapper import OpencvWrapper

def random_action(action_space):
    return action


if __name__ == '__main__':
    
    state_dim = 10
    action_dim = 5
    discount_factor = 1e-2
    max_step = 3
    neighbor_width = 1
    learning_rate = 1e-3
    
    observation_model = tf.random.uniform([1, state_dim, 
                                           state_dim, 1],
                                           dtype=tf.dtypes.float64)
    
    
    belief = tf.random.uniform([1, state_dim, 
                                state_dim, 1],
                                dtype=tf.dtypes.float64)
    
    target_action = tf.constant([1, 0, 0, 1, 0], dtype=tf.dtypes.float64, shape=[action_dim])
    
    agent = QMDP_RCNN(state_dim, action_dim, belief, target_action, 
                 transition_model, observation_model, target_belief, 
                 discount_factor, max_step, neighbor_width, learning_rate)
    
    agent.run()

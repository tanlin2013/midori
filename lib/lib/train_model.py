import numpy as np
import tensorflow as tf 

class ValueIterationRCNN:
    """
    Args:
        state_dim
        action_dim
        transition_model
        reward_function
        discount_factor
        max_step
        neighbor_width
        
    Attributes:
        run
        value_function
        
    Reference:
        arXiv: 1701.02392
    """
    def __init__(self, state_dim, action_dim, transition_model, 
                 reward_function, discount_factor, max_step, neighbor_width):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.transition_model = transition_model
        self.reward_function = reward_function
        self.discount_factor = discount_factor
        self.max_step = max_step
        self.neighbor_width = neighbor_width          
        self.value_function = tf.random.uniform([1, self.state_dim, 
                                                 self.state_dim, 1],
                                                dtype=tf.dtypes.float64)
        self.Q_value = None
        
    def _flipped_transition_model(self):
        N_t = 2*self.neighbor_width + 1
        begin = [N_t-self.neighbor_width, N_t-self.neighbor_width, 0, 0]
        size = [2*self.neighbor_width+1, 2*self.neighbor_width+1, 1, self.action_dim]
        reduced_transition_model = tf.slice(self.transition_model, begin, size)
        flipped_transition_model = tf.reverse(reduced_transition_model, [0])
        flipped_transition_model = tf.reverse(reduced_transition_model, [1])
        return flipped_transition_model
        
    def _build_model(self):
        for recurrence in range(self.max_step):
            conv = tf.nn.conv2d(self.value_function,
                                filter=self._flipped_transition_model(),
                                padding="SAME")
            
            conv *= self.discount_factor
            self.Q_value = tf.math.add(conv, self.reward_function)
            self.value_function = tf.nn.max_pool2d(self.Q_value, 
                                                   ksize=[1, 1, 1, self.action_dim],
                                                   strides=[1, 1, 1, self.action_dim],
                                                   padding="SAME")
            #print(self.value_function, self.Q_value)
        return self.value_function
            
    def run(self, sess=None):
        if not sess:
            sess = tf.Session()
        sess.run(self._build_model())
        return
         
    
class BeliefPropagationRCNN:  
    
    def __init__(self, state_dim, action_dim, choice, observation_model, 
                 belief, target_belief, max_step, neighbor_width, learning_rate):
        """
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.choice = choice
        self.observation_model = observation_model
        self.transition_model = tf.Variable(tf.random.uniform([self.state_dim, self.state_dim, 
                                                               1, self.action_dim],                                
                                                                0.0, 1.0,
                                                                dtype=tf.dtypes.float64),
                                            constraint=lambda t: tf.clip_by_value(t, 0, 1))
        self.belief = belief
        self.target_belief = target_belief
        self.max_step = max_step
        self.learning_rate = learning_rate
        self.initialize_variables = tf.initialize_all_variables()
    
    def _build_model(self):
        trans_begin = [0, 0, 0, self.choice]
        trans_size = [self.state_dim, self.state_dim, 1, 1]
        obs_begin = [0, 0, 0, self.choice]
        obs_size = [1, self.state_dim, self.state_dim, 1]
        for recurrence in range(self.max_step):
            conv = tf.nn.conv2d(self.target_belief,
                                filter=tf.slice(self.transition_model, trans_begin, trans_size),
                                padding="SAME")            
            
            self.belief = tf.math.multiply(conv, tf.slice(self.observation_model, obs_begin, obs_size))
            self.belief /= tf.norm(self.belief)
            #print(self.belief)
        return self.belief
    
    def _train_model(self):
        # TODO: add constrain for the domain of transition model
        loss = tf.losses.mean_squared_error(self._build_model(),
                                            self.target_belief,
                                            scope=None,
                                            loss_collection=tf.GraphKeys.LOSSES,
                                            reduction=tf.losses.Reduction.SUM)
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)
        return optimizer, loss
    
    def run(self, sess=None):
        if not sess: 
            sess = tf.Session()
        #sess.run(self.initialize_variables)
        sess.run(self._build_model())
        print("In entry BP RCNN")
        for epoch in range(self.max_step):
            _, loss = sess.run(self._train_model())
            print('epoch %d, loss=' %epoch, loss)
        return
        
    
class QMDP_RCNN:
    
    def __init__(self, state_dim, action_dim, belief, action_choices, observations,
                 observation_model, target_belief, discount_factor, 
                 max_step, neighbor_width, learning_rate):
        """
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.Q_value = tf.placeholder(dtype=tf.dtypes.float64, 
                                      shape=[1, self.state_dim, 
                                             self.state_dim, 1])
        self.belief = belief
        self.action_choices = action_choices
        self.observations = observations
        self.action = tf.placeholder(dtype=tf.dtypes.float64, shape=[self.action_dim])
        self.reward_function = tf.Variable(tf.random.uniform([1, self.state_dim, 
                                                              self.state_dim, self.action_dim], dtype=tf.dtypes.float64))
        self.transition_model = tf.placeholder(dtype=tf.dtypes.float64, 
                                      shape=[self.state_dim, self.state_dim,
                                             1, self.action_dim])
        self.observation_model = observation_model
        # TODO
        self.target_belief = target_belief
        self.discount_factor = discount_factor
        self.max_step = max_step
        self.neighbor_width = neighbor_width
        self.learning_rate = learning_rate
        self.initialize_variables = tf.initialize_all_variables()
    
    def _target_action(self):
        pass
    
    def _build_model(self, sess):
        choise = 2
        BP_RCNN = BeliefPropagationRCNN(self.state_dim, self.action_dim, choise, self.observation_model, 
                 self.belief, self.target_belief, self.max_step, self.neighbor_width, self.learning_rate)
        sess.run(BP_RCNN.initialize_variables)
        BP_RCNN._build_model()
        self.belief = BP_RCNN.belief
        self.transition_model = BP_RCNN.transition_model
        
        VI_RCNN = ValueIterationRCNN(self.state_dim, self.action_dim, self.transition_model, 
                 self.reward_function, self.discount_factor, self.max_step, self.neighbor_width)
        VI_RCNN._build_model()
        self.Q_value = VI_RCNN.Q_value
        
        #print(self.belief, self.Q_value)
        belief_Q_value = tf.tensordot(self.belief, self.Q_value, axes=[[1, 2], [1, 2]])
        #print(belief_Q_value)
        self.action = tf.reshape(tf.nn.softmax(belief_Q_value), shape=[-1])
        print(self.action, self.action.eval())
        return self.action
        
    def _train_model(self, sess):
        loss = tf.losses.sigmoid_cross_entropy(
                self.target_action,
                self._build_model(sess),
                weights=1.0,
                label_smoothing=0,
                scope=None,
                loss_collection=tf.GraphKeys.LOSSES,
                reduction=tf.losses.Reduction.SUM)     
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)
        return optimizer, loss
    
    def run(self):
        with tf.Session() as sess:
            sess.run(self.initialize_variables)
            #print(self.reward_function.eval())
            sess.run(self._build_model(sess))
            for epoch in range(self.max_step):
                _, loss = sess.run(self._train_model(sess))
                print('epoch %d, loss=' %epoch, loss)
        return
        
    
if __name__ == '__main__':
    
    state_dim = 10
    action_dim = 5
    transition_model = tf.random.uniform([state_dim, state_dim, 
                                          1, action_dim],
                                          dtype=tf.dtypes.float64)
    reward_function = tf.random.uniform([1, state_dim, 
                                         state_dim, action_dim],
                                         dtype=tf.dtypes.float64)
    discount_factor = 1e-2
    max_step = 10
    neighbor_width = 1
    
# =============================================================================
#     agent = ValueIterationRCNN(state_dim, action_dim, transition_model, 
#                  reward_function, discount_factor, max_step, neighbor_width)
#     agent.run()
# =============================================================================

    observation_model = tf.random.uniform([1, state_dim, 
                                           state_dim, action_dim],
                                           dtype=tf.dtypes.float64)
    belief = tf.random.uniform([1, state_dim, 
                                state_dim, 1],
                                dtype=tf.dtypes.float64)
    target_belief = tf.random.uniform([1, state_dim, 
                                       state_dim, 1],
                                       dtype=tf.dtypes.float64)
    learning_rate = 1e-3
    choice = 2
    
# =============================================================================
#     agent = BeliefPropagationRCNN(state_dim, action_dim, choice, observation_model, 
#                  belief, target_belief, max_step, neighbor_width, 
#                  learning_rate)
#     agent.run()
# =============================================================================
    
    target_action = tf.constant([1, 0, 0, 1, 0], dtype=tf.dtypes.float64, shape=[action_dim])
    
    action_choices = []
    observations = []
    
    agent = QMDP_RCNN(state_dim, action_dim, belief, action_choices, observations,
                 observation_model, target_belief, discount_factor, 
                 max_step, neighbor_width, learning_rate, target_action)
    
    agent.run()

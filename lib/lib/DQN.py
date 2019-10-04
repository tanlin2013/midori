# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np

class Net(nn.Module):
    
    def __init__(self, n_states, n_actions):
        super(Net, self).__init__()
        self.fcl = nn.Linear(n_states, 50)
        self.fcl.weight.data.normal_(0, 0.1)   # initialize fully connected layer
        self.out = nn.Linear(50, n_actions)
        self.out.weight.data.normal_(0, 0.1)   # initialize output layer

    def forward(self, x):
        x = self.fcl(x)
        x = nn.functional.relu(x)
        actions_value = self.out(x)
        return actions_value

class DQN(object):
    
    def __init__(self, n_states, n_actions, action_space_shape, hyper_params):
        self.n_states = n_states
        self.n_actions = n_actions
        self.action_space_shape = action_space_shape
        self.eval_net, self.target_net = Net(n_states, n_actions), Net(n_states, n_actions)
        self.hyper_params = hyper_params
        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0 # for storing memory
        self.memory = np.zeros((hyper_params["mem_capacity"], n_states * 2 + 2)) # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=hyper_params["learning_rate"])
        self.loss_func = nn.MSELoss()
        
    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < self.hyper_params["epsilon"]:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if self.action_space_shape == 0 else action.reshape(self.action_space_shape)  # return the argmax index
        else:   # random
            action = np.random.randint(0, self.n_actions)
            action = action if self.action_space_shape == 0 else action.reshape(self.action_space_shape)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.hyper_params["mem_capacity"]
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % self.hyper_params["target_update_freq"] == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(self.hyper_params["mem_capacity"], self.hyper_params["batch_size"])
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.n_states])
        b_a = torch.LongTensor(b_memory[:, self.n_states:self.n_states+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.n_states+1:self.n_states+2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.n_states:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + self.hyper_params["reward_discount"] * q_next.max(1)[0].view(self.hyper_params["batch_size"], 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
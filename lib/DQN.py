# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from lib.replay_memory import Transition, ReplayMemory


class Net(nn.Module):

    def __init__(self, h, w, outputs):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(16)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 16
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
# =============================================================================
#         x = nn.functional.relu(self.bn1(self.conv1(x)))
#         x = nn.functional.relu(self.bn2(self.conv2(x)))
#         x = nn.functional.relu(self.bn3(self.conv3(x)))
# =============================================================================
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = self.bn3(self.conv3(x))
        x = torch.sigmoid(x)
        return self.head(x.view(x.size(0), -1))


class DQN:
    
    def __init__(self, screen_height, screen_width, n_actions):
        self.n_actions = n_actions
        self.batch_size = 2
        self.gamma = 0.9
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 200
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = Net(screen_height, screen_width, n_actions).to(self.device)
        self.target_net = Net(screen_height, screen_width, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        #self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)

        self.steps_done = 0

    def select_action(self, state):
        sample = np.random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            np.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[np.random.randint(1, self.n_actions)]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        # loss = nn.functional.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.item()
        
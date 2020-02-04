import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from itertools import count
import os, sys
sys.path.insert(0, os.path.abspath('../'))
from lib.DQN import DQN
from envs.recaptcha import Recaptcha
import matplotlib
import matplotlib.pyplot as plt
from torchsummary import summary


class GuiAgent:

    def __init__(self):
        self.env = Recaptcha()
        screen_height, screen_width = self.env.resolution
        n_actions = self.env.action_space.n
        self.agent = DQN(screen_height, screen_width, n_actions)
        self.DQN_params = {}
        self.n_episodes = 50
        self.episode_durations = []
    
    def plot_durations(self):
        # set up matplotlib
        is_ipython = 'inline' in matplotlib.get_backend()
        if is_ipython:
            from IPython import display
        plt.ion()
        
        plt.figure(2)
        plt.clf()
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())
        return
    
    def get_screen(self, position):
        I = np.zeros(self.env.resolution)
        I[position[0]][position[1]] = 1
        
        # Convert to float, rescale, convert to torch tensor (CHW)
        # (this doesn't require a copy)
        screen_height, screen_width = self.env.resolution
        screen = I.reshape(1, screen_height, screen_width)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        resize = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])
                    #transforms.Resize(40, interpolation=Image.CUBIC),
                    
        return resize(screen).unsqueeze(0).to(self.agent.device)
    
    def render(self):
        for i_episode in range(self.n_episodes):
            # Initialize the environment and state
            self.env.reset()
            state = self.get_screen(self.env.state)
            for t in count():
                # Select and perform an action
                action = self.agent.select_action(state)
                _, done, reward = self.env.step(action.item())
                print(t, reward)
                reward = torch.tensor([reward], device=self.agent.device)
                
                # Observe new state
                if not done:
                    next_state = self.get_screen(self.env.state)
                else:
                    next_state = None

                # Store the transition in memory
                self.agent.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                self.agent.optimize_model()
                if done:
                    self.episode_durations.append(t + 1)
                    self.plot_durations()
                    break
            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.agent.target_update == 0:
                self.agent.target_net.load_state_dict(self.agent.policy_net.state_dict())

        print('Complete')
        plt.ioff()
        plt.show()


if __name__ == '__main__':

    agent = GuiAgent()
    agent.render()

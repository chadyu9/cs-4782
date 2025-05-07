import numpy as np
import random
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


#---------------------------------------#
#   UTILITIES FOR YOU * DO NOT MODIFY   #
#---------------------------------------#

class Transition(object):
  def __init__(self, s, a, next_s, r):
    self.state = s
    self.action = a
    self.next_state = next_s
    self.reward = r


class ExperienceReplay(object):
    def __init__(self, capacity):

        # Capacity of the agent's memory, always an int
        self.cap = capacity
        # A list of Transition objects that represent the agent's experiences
        self.memory = []
        # Current position of the agent, 0 <= self.pos < len(self.memory)
        self.pos = 0


    def push(self, *args):
        """
        Push a transition to the agent's memory.
        """
        if len(self.memory) < self.cap:
            self.memory.append(None)
        self.memory[self.pos] = Transition(*args)
        self.pos = (self.pos + 1) % self.cap


    def sample(self, batch_size):
        """
        Uniformly random sample batch_size of Transitions
        """
        return random.sample(self.memory, batch_size)


    def __len__(self):
        return len(self.memory)
    

#--------------------------------#
#   Q-Learning (TODOs: #1,2,3)   # 
#--------------------------------#

def sample_action(Q, s, env, epsilon):
    """
    Sample an action using epsilon-greedy policy.

    Q: matrix of size (states_size, action_space_size)
    s: current state
    env: the gym environment (hint: you can sample a random action from the env)
    epsilon: exploration rate
    """
    ### TODO 1:
    # Return action index from Q table with epsilon-greedy exploration.
    pass

def q_update(Q, s, a, reward, s_next, alpha, gamma):
    """
    Update the Q table using the Q-learning formula.

    Q: matrix of size (states_size, action_space_size)
    s: current state
    a: action taken
    reward: reward received
    s_next: next state
    alpha: learning rate
    gamma: discount factor
    """
    ### TODO 2:
    # Return updated Q-Table with new knowledge
    pass

def q_learning(Q, env, epsilon, alpha, gamma, epis):
    """
    Returns the final Q table and the rewards per episode.

    Q: matrix of size (states_size, action_space_size)
    env: the environment
    epsilon: exploration rate
    alpha: learning rate
    gamma: discount factor
    epis: number of episodes
    """
    # NOTE: Keep track of the rewards obtained over an episode (aka. each trajectory).
    rewards_per_episode = []

    for i in range(epis):
        ### TODO 3:
        # Complete the Q-learning algorithm by filling in the TODOs.
        # You can reuse the code from the q_update and sample_action.

        # TODO: Reset environment

        # NOTE: An episode is one iteration of the while loop (one trajectory)
        while True:

            # TODO: Choose action from Q table with epsilon-greedy exploration.

            # TODO: Get new state & reward from environment

            # TODO: Update Q-Table with new knowledge

            # TODO: Update state

            # TODO: If terminated or truncated, break out of the loop
                break
        # TODO: Append total reward for the episode

        # Print progress every 100 episodes
        if (i + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"Episode {i + 1}/{epis} | Average Reward (last 100 episodes): {avg_reward:.2f}")

    return Q, rewards_per_episode


#-----------------------------------#
#   Deep Q-Networks (TODOs: #4,5)   # 
#-----------------------------------#

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        """
        Parameters:
            n_observations (int): Number of observations (input features).
            n_actions (int): Number of actions (output features).
        """
        super(DQN, self).__init__()

        ### TODO 4: 
        # Define the linear layers

        pass

    def forward(self, x):

        ### TODO 5: 
        # Define the forward pass

        pass
        
        
#-------------------------------#
#   DQN Agent (TODOs: #6,7,8)   # 
#-------------------------------# 
        
class DQNAgent:
    def __init__(self, env, device, memory_capacity=10000, batch_size=128, gamma=0.99,
                 epsilon_start=0.9, epsilon_end=0.08, epsilon_decay=1000,
                 target_update=10, lr=1e-3):
        
        self.device = device

        # Hyperparameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = epsilon_start
        self.eps_end = epsilon_end
        self.eps_decay = epsilon_decay
        self.target_update = target_update
        self.lr = lr

        state, _ = env.reset()
        self.n_actions = env.action_space.n
        self.n_observations = len(state)
        self.steps_done = 0

        self.policy_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net = DQN(self.n_observations, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.memory = ExperienceReplay(memory_capacity)

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr)
        self.criterion = nn.SmoothL1Loss()


    def select_action(self, state):
        """
        Choose an action based on the current state using decayed ε-greedy strategy
        """
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        ### TODO 6: 
        # Implement the epsilon-greedy strategy with the eps_threshold
        # Remember to wrap the result in a (1,1) tensor and move it to the correct device.

        pass

    def get_values(self):
        """
        Returns the states, actions, and rewards from the sampled batch of transitions in memory.

        The function performs the following steps:
        1. Checks if the number of transitions in memory is less than the batch size. If so, it exits early.
        2. Samples a batch of transitions from memory.
        3. Extracts states, next states, actions, and rewards from the sampled transitions.
        4. Creates a mask to identify non-final next states (states that are not None).
        5. Collates all non-final next states, states, actions, and rewards into their respective batches.

        Returns:
            tuple: A tuple containing the following elements:
            - Q-values of the current states based on the policy network (Tensor).
            - Batch of rewards (Tensor).
            - Mask identifying non-final next states (Boolean Tensor).
            - Batch of non-final next states (Tensor).
        """
        if len(self.memory) < self.batch_size:
            return None
        batch = self.memory.sample(self.batch_size)

        next_state = []
        state = []
        action = []
        reward = []
        for x in batch:
          next_state.append(x.next_state)
          state.append(x.state)
          action.append(x.action)
          reward.append(x.reward)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in next_state if s is not None])

        state_batch = torch.cat(state)
        action_batch = torch.cat(action)
        reward_batch = torch.cat(reward)

        return self.policy_net(state_batch).gather(1, action_batch), reward_batch, non_final_mask, non_final_next_states


    def optimize_model(self):
        if self.get_values() is not None:
            state_action_values, reward_batch, non_final_mask, non_final_next_states = self.get_values()
        else:
            return

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

        ### TODO 7: 
        # Compute the target state_action_values with the Bellman Equation

        ### TODO 8: 
        # Compute the loss between the predicted state_action_values and the target state_action_values using the SmoothL1Loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1)
        self.optimizer.step()
        
        
#-----------------------------------#
#   Policy Network (TODOs: #9,10)   #
#-----------------------------------#

class PolicyNetwork(nn.Module):
    def __init__(self, number_observation_features: int, number_actions: int):
        """Initialize the PolicyNetwork with an input layer, one hidden layer, and an output layer.

        Args:
            number_observation_features (int): Number of features in the observation space.
            number_actions (int): Number of possible actions.
        """
        super(PolicyNetwork, self).__init__()

        self.hidden_layer_features = 128

        ### TODO 9: 
        # Define the linear layers and non-linearities

    def forward(self, x):
        """Forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor containing the observation.

        Returns:
            torch.Tensor: The output tensor containing the action scores.
        """
        ### TODO 10: 
        # Pass through the linear layers and non-linearities

        pass 
        

#----------------------------------------#
#   Reinforce Agent (TODOs: #11,12,13)   #
#----------------------------------------#
from torch.distributions import Categorical

class ReinforceAgent:
    def __init__(self, env):
        state, _ = env.reset()
        self.model = PolicyNetwork(len(state), env.action_space.n)
        self.optimizer = optim.AdamW(self.model.parameters(), 1e-3)


    def choose_action(self, state):
        """
        Returns the action to take given the current state and the log probability of that action.

        Parameter:
            state: the current state of the agent

        Returns:
            tuple: A tuple containing the following elements:
            - Chosen action (integer): The specific action selected to be taken in the environment.
            - Log probability (Tensor): The logarithm of the probability of the chosen action according to the policy.

        """
        ### TODO 11: 
        # Get action distribution from policy
        # Model the action distribution with a categorical one for sampling 
        #   (c.f. Categorical from torch.distributions)
        # Sample an action and return a tuple of it and its logprob
        ###

        pass

    def calculate_reinforce_loss(self, log_probs, returns):
        """
        Calculates the REINFORCE loss with a baseline to reduce variance.

        Parameters:
            log_probs (Tensor): A tensor containing the logarithm of the probabilities of actions that have been taken.
            returns (Tensor): A tensor containing the returns (cumulative rewards) obtained after taking the respective actions.

        The function performs the following steps:

        Returns:
            Tensor: The estimated policy gradient with baseline.
        """
        ### TODO 12: 
        # Subtract the mean from the returns to obtain the "advantage" values
        # Multiply the negative log probs with the advantage to obtain per-action losses
        # Return the mean loss over actions
        ###

        pass


def calculate_discounted_returns(episode_rewards, gamma):
    """
    Calculate the discounted returns for an episode.

    Parameters:
        episode_rewards (list of float): A list of rewards obtained during an episode.
        gamma (float): The discount factor for future rewards (between 0 and 1).

    Returns:
        discounted_returns (list of float): A list of discounted returns for each time step in the episode.
    """
    discounted_returns = []
    discounted_return = 0

    ### TODO 13:
    # Be careful to follow the instructions in the notebook. 
    # It's possible to get this wrong but still return a learnable signal.

    pass


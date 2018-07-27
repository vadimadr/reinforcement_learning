import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from common import obesrvation_shape, device, Agent
from utils import get_cumulative_rewards, generate_session, generate_session_batch, get_total_rewards


def reinforce_loss(logprobas, actions, rewards):
    # select log-probabilities for chosen actions, log pi(a_i|s_i)
    logprobas_for_actions = torch.sum(logprobas * to_one_hot(actions).to(torch.float), dim=1)
    return torch.mean(-logprobas_for_actions * rewards)


class ActorCritic(Agent):
    def __init__(self, observation_space, action_space, args):
        self.model = MLPPolicy(observation_space, action_space).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), args.lr)
        self.gamma = args.gamma
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef

    def save(self, path):
        torch.save({
            'state_dict': self.model.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        state_dict = checkpoint.get('state_dict', checkpoint)
        self.model.load_state_dict(state_dict)

    def update(self, states, actions, cumulative_rewards, dones=None):
        states = torch.tensor(states).to(device, torch.float)
        actions = torch.tensor(actions).to(device, torch.float)
        cumulative_returns = torch.tensor(cumulative_rewards).to(device, torch.float)

        # predict logits, probas and log-probas using an agent.
        logits, values = self.model(states)
        probas = F.softmax(logits, dim=1)
        logprobas = F.log_softmax(logits, dim=1)

        # REINFORCE objective function
        self.optimizer.zero_grad()
        rewards_with_baseline = cumulative_returns - values.squeeze(1)
        policy_loss = reinforce_loss(logprobas, actions, rewards_with_baseline)
        value_loss = F.smooth_l1_loss(values, cumulative_returns.unsqueeze(1))
        entropy = torch.mean(-torch.sum(logprobas * probas, dim=1))
        loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy
        loss.backward()
        self.optimizer.step()

        return loss, policy_loss, value_loss, entropy

    def get_action(self, state):
        with torch.no_grad():
            logits, _ = self.model(torch.tensor(state).to(device, torch.float))
            dist = Categorical(F.softmax(logits, dim=1))
            sample = dist.sample()
            return sample.detach().cpu().numpy()

    def get_value(self, state):
        with torch.no_grad():
            _, values = self.model(torch.tensor(state).to(device, torch.float))
            return values.detach().cpu().numpy()


class MLPPolicy(nn.Module):
    def __init__(self, observation_space, action_space):
        super(MLPPolicy, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(*obesrvation_shape(observation_space), 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.logits = nn.Linear(64, action_space.n)
        self.state_values = nn.Linear(64, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.model(x)
        return self.logits(x), self.state_values(x)


def to_one_hot(y, n_dims=None):
    """ Take an integer vector (tensor of variable) and convert it to 1-hot matrix. """
    y_tensor = y.to(torch.long).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = y_tensor.new_zeros(y_tensor.size(0), n_dims).scatter_(1, y_tensor, 1)
    return y_one_hot


def train(agent, env, args, max_reward=200):
    total_rewars = []
    for i in range(args.num_steps):
        states, actions, rewards, dones, lastvalues = generate_session_batch(agent, env, args.episode_length)

        # get cumulative rewards
        # use fictive reward for handling unfinished sessions
        rewards_ = np.c_[rewards, lastvalues]
        dones_ = np.c_[dones, np.zeros(dones.shape[0])]
        cumulative_rewards = get_cumulative_rewards(rewards_, agent.gamma, dones=dones_)[:, :-1]

        # reshape sessions to form a batch
        batch_size = env.num_envs * args.episode_length
        states = states.reshape((batch_size,) + env.observation_space.shape)
        actions = actions.reshape((batch_size,))
        cumulative_rewards = cumulative_rewards.reshape((batch_size,))
        dones = dones.reshape((batch_size,))

        # update policy on a batch
        losses = agent.update(states, actions, cumulative_rewards, dones)
        # note that last session reward is truncated
        total_rewars.extend(get_total_rewards(rewards, dones))

        current_mean_reward = np.mean(total_rewars[-100:])
        if i % 100 == 0:
            print("Iteration: %i, Mean reward:%.3f" % (i, current_mean_reward))
            if current_mean_reward > 200:
                return

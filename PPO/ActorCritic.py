# copied from https://github.com/geekyutao/PyTorch-PPO/blob/master/PPO_discrete.py
# slight modifications for challenge 2

import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def act(self, state, memory, deterministic=False, full=False):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        # return list of probs
        if full:
            return action_probs

        # for training
        if not deterministic:
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(action_logprob)
            return action.item()

        # no sense following deterministic policy during training, so no memory needed
        else:
            max_actions = torch.argmax(action_probs, dim=1)
            return max_actions


    def evaluate(self, state, action):
        state_value = self.critic(state)

        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, torch.squeeze(state_value), dist_entropy
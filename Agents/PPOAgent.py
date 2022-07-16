# copied from https://github.com/geekyutao/PyTorch-PPO/blob/master/PPO_discrete.py
# only changes involve keeping track of decoys, adding scanning states, and reduction of action space

from PPO.ActorCritic import ActorCritic
from PPO.Memory import Memory
import torch
import torch.nn as nn
from CybORG.Agents import BaseAgent
import numpy as np
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PPOAgent(BaseAgent):
    def __init__(self, input_dims=52, action_space=[i for i in range(158)], lr=0.002, betas=[0.9, 0.990], gamma=0.99, K_epochs=4, eps_clip=0.2, restore=False, ckpt=None,
                 deterministic=False, training=True, start_actions=[]):

        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.input_dims = input_dims
        self.restore = restore
        self.ckpt = ckpt
        self.deterministic = deterministic
        self.training = training
        self.start = start_actions

        # reset decoys
        self.end_episode()
        # initialise
        self.set_initial_values(action_space=action_space)

    # add a decoy to the decoy list
    def add_decoy(self, id, host):
        # add to list of decoy actions
        if id not in self.current_decoys[host]:
            self.current_decoys[host].append(id)

    # remove a decoy from the decoy list
    def remove_decoy(self, id, host):
        # remove from decoy actions
        if id in self.current_decoys[host]:
            self.current_decoys[host].remove(id)

    # add scan information
    def add_scan(self, observation):
        indices = [0, 4, 8, 12, 28, 32, 36, 40, 44, 48]
        for id, index in enumerate(indices):
            # if scan seen on defender, enterprise 0-2, opserver0 or user 0-4
            if observation[index] == 1 and observation[index+1] == 0:
                # 1 if scanned before, 2 if is the latest scan
                self.scan_state = [1 if x == 2 else x for x in self.scan_state]
                self.scan_state[id] = 2
                break

    # concatenate the observation with the scan state
    def pad_observation(self, observation, old=False):
        if old:
            # added for store transition, remnants of DQN
            return np.concatenate((observation, self.scan_state_old))
        else:
            return np.concatenate((observation, self.scan_state))


    def get_action(self, observation, action_space=None):
        # not needed for ppo since no transitions (remnant of DQNAgent)
        self.scan_state_old = copy.copy(self.scan_state)

        self.add_scan(observation)
        observation = self.pad_observation(observation)
        state = torch.FloatTensor(observation.reshape(1, -1)).to(device)
        action = self.old_policy.act(state, self.memory, deterministic=self.deterministic)
        action_ = self.action_space[action]


        # force start actions, ignore policy. only for training
        if len(self.start_actions) > 0:
            action_ = self.start_actions[0]
            self.start_actions = self.start_actions[1:]


        if action_ in self.decoy_ids:
            host = action_
            # select a decoy from available ones
            action_ = self.select_decoy(host, observation=observation)

        # if action is a restore, delete all decoys from decoy list for that host
        if action_ in self.restore_decoy_mapping.keys():
            for decoy in self.restore_decoy_mapping[action_]:
                for host in self.decoy_ids:
                    if decoy in self.current_decoys[host]:
                        self.remove_decoy(decoy, host)

        return action_

    def store(self, reward, done):
        self.memory.rewards.append(reward)
        self.memory.is_terminals.append(done)

    def clear_memory(self):
        self.memory.clear_memory()

    def select_decoy(self, host, observation):
        try:
            # pick the top remaining decoy
            action = [a for a in self.greedy_decoys[host] if a not in self.current_decoys[host]][0]
            self.add_decoy(action, host)
        except:
            # # otherwise just use the remove action on that host
            # action = self.host_to_remove[host]

            # pick the top decoy again (a non-action)
            if self.training:
                action = self.greedy_decoys[host][0]

            # pick the next best available action (deterministic)
            else:
                state = torch.FloatTensor(observation.reshape(1, -1)).to(device)
                actions = self.old_policy.act(state, self.memory, full=True)

                max_actions = torch.sort(actions, dim=1, descending=True)
                max_actions = max_actions.indices
                max_actions = max_actions.tolist()

                # don't need top action since already know it can't be used (hence could put [1:] here, left for clarity)
                for action_ in max_actions[0]:
                    a = self.action_space[action_]
                    # if next best action is decoy, check if its full also
                    if a in self.current_decoys.keys():
                        if len(self.current_decoys[a]) < len(self.greedy_decoys[a]):
                            action = self.select_decoy(a,observation)
                            self.add_decoy(action, a)
                            break
                    else:
                        # don't select a next best action if "restore", likely too aggressive for 30-50 episodes
                        if a not in self.restore_decoy_mapping.keys():
                            action = a
                            break
        return action

    def train(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        old_states = torch.squeeze(torch.stack(self.memory.states).to(device)).detach()
        old_actions = torch.squeeze(torch.stack(self.memory.actions).to(device)).detach()
        old_logprobs = torch.squeeze(torch.stack(self.memory.logprobs)).to(device).detach()

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            ratios = torch.exp(logprobs - old_logprobs.detach())

            advantages = rewards - state_values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = - torch.min(surr1, surr2)

            critic_loss = 0.5 * self.MSE_loss(rewards, state_values) - 0.01 * dist_entropy

            loss = actor_loss + critic_loss

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.old_policy.load_state_dict(self.policy.state_dict())

    def end_episode(self):
        # 9 possible decoys: enterprise 0-2 and user 1-4, defender, opserver0 (cant do actions on user0)
        self.current_decoys = {1000: [], # enterprise0
                               1001: [], # enterprise1
                               1002: [], # enterprise2
                               1003: [], # user1
                               1004: [], # user2
                               1005: [], # user3
                               1006: [], # user4
                               1007: [], # defender
                               1008: []} # opserver0
        # 10 possible scans: defender, enterprise 0-2, user 0-4, opserver
        self.scan_state = np.zeros(10)
        # remnants of DQNAgent for store_transitions
        self.scan_state_old = np.zeros(10)
        # add start actions
        self.start_actions = copy.copy(self.start)


    def set_initial_values(self, action_space, observation=None):
        self.memory = Memory()

        self.greedy_decoys = {1000: [55, 107, 120, 29],  # enterprise0 decoy actions
                              1001: [43],  # enterprise1 decoy actions
                              1002: [44],  # enterprise2 decoy actions
                              1003: [37, 115, 76, 102],  # user1 decoy actions
                              1004: [51, 116, 38, 90],  # user2 decoy actions
                              1005: [130, 91],  # user3 decoy actions
                              1006: [131],  # user4 decoys
                              1007: [54, 106, 28, 119], # defender decoys
                              1008: [61, 35, 113, 126]} # opserver0 decoys

        # added to simplify / for clarity
        self.all_decoys = {55: 1000, 107: 1000, 120: 1000, 29: 1000,
                           43: 1001,
                           44: 1002,
                           37: 1003, 115: 1003, 76: 1003, 102: 1003,
                           51: 1004, 116: 1004, 38: 1004, 90: 1004,
                           130: 1005, 91: 1005,
                           131: 1006,
                           54: 1007, 106: 1007, 28: 1007, 119: 1007,
                           126: 1008, 61: 1008, 113: 1008, 35: 1008}

        # # no longer needed (since default action on a full decoy will depend on self.training)
        # self.host_to_remove = {1000: 16,  # enterprise0 remove
        #                        1001: 17,  # enterprise1 remove
        #                        1002: 18,  # enterprise2 remove
        #                        1003: 24,  # user1 remove
        #                        1004: 25,  # user2 remove
        #                        1005: 26,  # user3 remove
        #                        1006: 27,  # user4 remove
        #                        1007: 15,  # defender remove
        #                        1008: 22}  # remove opserver0

        # make a mapping of restores to decoys
        self.restore_decoy_mapping = dict()
        # decoys for defender host
        base_list = [28, 41, 54, 67, 80, 93, 106, 119]
        # add for all hosts
        for i in range(13):
            self.restore_decoy_mapping[132 + i] = [x + i for x in base_list]

        # we statically add 9 decoy actions
        action_space_size = len(action_space)
        self.n_actions = action_space_size + 9
        self.decoy_ids = list(range(1000, 1009))

        # add decoys to action space (all except user0)
        self.action_space = action_space + self.decoy_ids

        # add 10 to input_dims for the scanning state
        self.input_dims += 10


        self.policy = ActorCritic(self.input_dims, self.n_actions).to(device)
        if self.restore:
            pretained_model = torch.load(self.ckpt, map_location=lambda storage, loc: storage)
            self.policy.load_state_dict(pretained_model)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)

        self.old_policy = ActorCritic(self.input_dims, self.n_actions).to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())

        self.MSE_loss = nn.MSELoss()
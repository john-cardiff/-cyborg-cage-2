# Attempt 1

We began our experimentations with DDQN.

In our first attempt we sought to create strong benchmarks to beat. Given that the Restore action removes decoys, we needed to keep track of them in our blue agent's implementation. This was achieved by maintaining a decoy list and then appending and removing from it at each decoy or relevant Restore action taken.

    # add a decoy to the decoy list
    def add_decoy(self, id):
        # add to list of decoy actions
        if id not in self.decoys_list:
            self.decoys_list.append(id)

    # remove a decoy from the decoy list, happens if the host gets restored
    def remove_decoy(self, id):
        # remove from decoy actions
        if id in self.decoys_list:
            self.decoys_list.remove(id)
            
In our first approach, we created a single new action for all decoys which would select from a list of greedy actions. In action selection it would select the "best" remaining decoy in the decoy list (therefore should a host get restored, the "best" remaining decoy may change).

    # select the top remaining decoy
    def select_decoy(self):
        try:
            action = [a for a in self.decoys if a not in self.decoys_list][0]
        except:
            # otherwise reselect the top decoy (useless action)
            action = self.decoys[0]
        return action


      
We performed a greedy search on both B_line and Meander until the Op_Server0 was "impacted" (averaging over 250 episodes at each step), and found the following list for B_line (while excluding decoys for Op_Server0). This became our decoy list for the agent.

      B_line: [44, 43, 29, 55, 107, 51, 37, 130, 115, 76, 116, 131, 120, 38, 91, 102, 90]
      
Finally, to set strong benchmarks and assist in exploration, we reduced the action set to the following. This was informed by our understanding of the challenge.

    action_space = [17,18] # remove enterprise 1 and 2
    action_space += [3,4,5,9] # analyse enterprise 0,1,2 and opserver
    action_space += [133,134,135,139] # restore enterprise 0,1,2 and opserver
    
Therefore the network chose from 11 actions: 10 non-decoy and 1 decoy (which selects greedily from the remaining available decoys, up to 17).

Given the limited action set, we opted for a relatively small number of episodes for the random hyperparamaters search in our DDQN implementation. We created 50 models for B_line and 50 for Meander with the following parameters while utilising the RMSprop optimiser.

    hyperparams = {"num_eps": [500],
                   "replace": [2500,5000],
                   "mem_size": [2500,5000],
                   "lr": [0.0005,0.00005,0.005],
                   "eps_dec": [0.00001,0.00002,0.000015],
                   "gamma": [0.95, 0.99, 0.9],
                   "batch_size": [16,32,64],
                   "eps_min": [0.01,0.05,0.1],
                   "eps_len": [100]}
                   
 The architecture was as follows for both models.

        self.fc1 = nn.Linear(input_dims, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_actions)

Where fc1 and fc2 had ReLU activations.
                

Our best result on B_line was around -16 and -49 for Meander (this considered changes to the BlueTableWrapper but not Host.py) on 1,000 episode evaluations.
            
# Attempt 2 (addressing flaw 1,2)

The results for B_line were quite good in our first attempt (even before the modifications to the BlueTableWrapper) and outperformed vanilla RL approaches. However, we felt that improvements could still be made given that there are several obvious flaws.
1. The approach fails to target decoys for specific hosts. This is problematic as the first few steps are on user hosts and not enterprise servers. Obviously, one could modify the greedy search to consider this (by utilising a Genetic Algorithm and evaluating the list based on a previously trained agent, i.e for the fitness function), but we wanted to discover a better method.
2. The buffer of decoys may fill, and then the blue action is wasted.
3. The blue agent is clueless on its current state (i.e start or end of an episode, which hosts have been scanned so far etc.) and therefore cannot target decoys accordingly.
4. The action space was considerably reduced through our intuition; this needed to be verified. 
5. DDQN is known to be unstable; therefore another algorithm may be a better choice (particularly if we expand the action set).

For the second attempt, we aimed to address the first two flaws. We built a new agent which instead of a single decoy action selecting greedily from all available decoys, had 9 new decoy actions which selected greedily from available decoys on their host. This addressed the first flaw.

        self.greedy_decoys = {1000: [55, 107, 120, 29],  # enterprise0 decoy actions
                              1001: [43],  # enterprise1 decoy actions
                              1002: [44],  # enterprise2 decoy actions
                              1003: [37, 115, 76, 102],  # user1 decoy actions
                              1004: [51, 116, 38, 90],  # user2 decoy actions
                              1005: [130, 91],  # user3 decoy actions
                              1006: [131],  # user4 decoys
                              1007: [54, 106, 28, 119], # defender decoys
                              1008: [61, 35, 113, 126]} # opserver0 decoys

Furthermore, if the buffer filled, it would pick the Remove action on that host. This would consider the second flaw.

    # select the top remaining decoy
    def select_decoy(self, host):
        try:
            # pick the top remaining decoy
            action = [a for a in self.greedy_decoys[host] if a not in self.current_decoys[host]][0]
        except:
            # otherwise just use the remove action on that host
            action = self.host_to_remove[host]
        return action


We utilised the same MLP architecture as Attempt 1, but the action set differed as we no longer required the Remove action.

    action_space = [133,134,135,139] # restore enterprise and opserver
    action_space += [11,12,13,14] # analyse user hosts
    # action_space += [24,25,26,27] # remove user hosts - remove not required
    action_space += [3,4,5,9] # analyse servers
    action_space += [141,142,143,144] # restore users
    # action_space += [17,18] # - remove not required
    

 Unfortunately, **no improvements were noted** despite longer training to consider the larger action set.
                   
# Attempt 3 (addressing flaw 3)

To address the third flaw, we wanted to add some memory to the "Attempt 2" agent by keeping track of scans.

3. The blue agent is clueless on its current state (i.e start or end of an episode, which hosts have been scanned so far etc.) and therefore cannot target decoys accordingly.

We maintained a list representing whether hosts had been scanned (value of 1) or not (value of 0) by implementing an add_scan method. Furthermore, the "latest" scan would have a value of 2 instead of 1 (this would allow the agent to keep track of recent activity).

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


We then padded the observation received by the agent with this list.

    # concatenate the observation with the scan state
    def pad_observation(self, observation, old=False):
        if old:
            # useful for store transitions in DDQN
            return np.concatenate((observation, self.scan_state_old))
        else:
            return np.concatenate((observation, self.scan_state))
            
 So in Attempt 3 we were keeping track of both the decoys and the scanning state.
 
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
        # useful for store transitions in DDQN
        self.scan_state_old = np.zeros(10)
        
During evaluation instead of defaulting to a non-action action, it would select the next best available action. So at testing time, if the next best action is to decoy a host already full of decoys, then it would continue in the list.

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

While utilising the same hyperparamaters and action space as Attempt 2, and by making the changes to Host.py, we achieved a top score of -27 on Meander (1,000 episodes of length 100). It should be noted that we observed improvements before evaluation (therefore the better results were not only achieved by an enhanced policy at testing time).


# Final attempt (addressing flaw 4,5)

We now aimed to address the 4th and 5th flaws.

4. The action space was considerably reduced through our intuition; this needed to be verified. 
5. DDQN is known to be unstable; therefore another algorithm may be a better choice (particularly if we expand the action set).

We expanded the action space to the following.

    action_space = [133, 134, 135, 139]  # restore enterprise and opserver
    action_space += [3, 4, 5, 9]  # analyse enterprise and opserver
    action_space += [16, 17, 18, 22]  # remove enterprise and opserer
    action_space += [11, 12, 13, 14]  # analyse user hosts
    action_space += [141, 142, 143, 144]  # restore user hosts
    action_space += [132]  # restore defender
    action_space += [2]  # analyse defender
    action_space += [15, 24, 25, 26, 27]  # remove defender and user hosts
    
And implemented a PPOAgent with the following architecture.

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
    
We also added our enhancements detailed in the first three Attempts to the PPO Agent. Recall that the changes were:

1. Having a single decoy action per host (which selects greedily from available decoys on that host), i.e 9 new actions
2. Padding the observation with the scanning state (including a "special" value of 2 for the most recent), i.e increasing the observation by 10

At testing time we selected the next best available action in the case where the decoy action was selected for a "full host". In addition, we turned the outputed probability distribution of PPO into a deterministic choice (by taking the argmax of the action vector). 

Now we had two good policies, but how could we put them together for our final agent? Fortunately the policies kind of agree on the first three steps (given that it is deterministic, we can find the actions). The first two steps are always two decoys: Fermitter User2, Tomcat User2. The third action is also always a decoy but depends on the second action of the red agent.

| Agent      | User0 scan | User1 scan | User2 scan | User3 scan | User4 scan |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | 
| B_line PPO     | N/A     | Fermitter Ent1    |  Fermitter Ent1    | SMTP Ent0 | SMTP Ent0 | 
| Meander PPO     | Apache User2 | SMTP Ent0 | SMTP Ent0 | Apache User2     | Apache User2      |  

For the fourth action we would know if we are facing Meander or B_line as the former can be fingerprinted by two consecutive scans (on different hosts), therefore we could assign the appropriate agent. Fermitter Ent1 and SMTP Ent0 for B_line are very close to equivalent, but the concern is mostly for Meander when User2 hasn't been scanned since it appears ideal to keep utilising decoys on it. However, for a User 1 scan it prefers SMTP Ent0 so the improvement is likely negligible. As a result, we retrained two agents where we forced the first three actions/decoys to be Fermitter User2, Tomcat User2 and SMTP Ent0.

# Results

We did some hyperparamater tuning, but it didn't seem to matter (so long as the buffer_size and the number of episodes were sufficiently large) whereby the models converged to about -13 for B_line 100 and -16 for Meander 100. Given that the training policy differed slightly from the evaluation policy, we analysed the results at different checkpoints on the evaluation policy (select next best action if decoy is full on that host + turn policy deterministic), and chose two of our models (unfortunately we did not have sufficient time to tune, instead we selected from several dozen models). As a side note interesting analysis could be performed here such as: covariance between the two policies when approaching "convergence", selecting the best next decoy instead of action, are there any "convergent" models where the stochastic policy is better than the deterministic one (mixed strategy?), verifying that models trained with the evaluation policy (stochastic one obviously) would lead to instability. 
        
Finally, we ran evaluation.py at random.seed(0) and achieved the following results for the MainAgent (which starts with the three actions described in the Final attempt, fingerprints the red agent and finally loads the blue agent).

        Terminal output:
        
        Average reward for red agent B_lineAgent and steps 30 is: -3.4108000000000005 with a standard deviation of 1.7701572858977288
        Average reward for red agent RedMeanderAgent and steps 30 is: -5.539 with a standard deviation of 1.2888600234936918
        Average reward for red agent SleepAgent and steps 30 is: 0.0 with a standard deviation of 0.0
        Average reward for red agent B_lineAgent and steps 50 is: -6.343799999999996 with a standard deviation of 2.4081654471445293
        Average reward for red agent RedMeanderAgent and steps 50 is: -8.8347 with a standard deviation of 2.2001469397648314
        Average reward for red agent SleepAgent and steps 50 is: 0.0 with a standard deviation of 0.0
        Average reward for red agent B_lineAgent and steps 100 is: -13.237799999999984 with a standard deviation of 4.254330792320992
        Average reward for red agent RedMeanderAgent and steps 100 is: -16.2844 with a standard deviation of 3.825688504378338
        Average reward for red agent SleepAgent and steps 100 is: 0.0 with a standard deviation of 0.0
        

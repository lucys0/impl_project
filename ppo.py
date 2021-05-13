import gym
import time

import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Normal
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

"""
    This PPO implementation follows the pseudocode provided in OpenAI's Spinning Up for PPO: 
    https://spinningup.openai.com/en/latest/algorithms/ppo.html. 
    Pseudocode line numbers are specified as "ALG STEP #" in ppo.py.
"""


class PPO:
    def __init__(self, policy_class, env, writer, device, encoder=None, **hyperparameters):
        """
            Initializes the PPO model, including hyperparameters.
            Parameters:
                policy_class - the policy class to use for our actor/critic networks.
                env - the environment to train on.
                hyperparameters - all extra arguments passed into PPO that should be hyperparameters.
            Returns:
                None
        """
        # Initialize hyperparameters for training with PPO
        self._init_hyperparameters(hyperparameters)

        # Extract environment information
        self.env = env
        # self.obs_dim = (
        # 	env.observation_space.shape[0]) * (env.observation_space.shape[1])
        # self.obs_dim = env.observation_space.shape[0]
        self.obs_dim = 32 * 27 * 27 # cnn
        self.act_dim = env.action_space.shape[0]
            
        # Set the encoder and writer
        self.encoder = encoder.to(device)
        self.writer = writer

        # Record timesteps taken
        self.t = 0
        
     # Initialize actor and critic networks
        # ALG STEP 1
        if self.encoder:
            self.actor = nn.Sequential(self.encoder, policy_class(self.obs_dim, self.act_dim, is_actor=True))
            self.critic = nn.Sequential(self.encoder, policy_class(self.obs_dim, 1, is_actor=False))
        else:
            self.actor = policy_class(self.obs_dim, self.act_dim)
            self.critic = policy_class(self.obs_dim, 1)
        self.actor.to(device)
        self.critic.to(device)

        # Initialize optimizers for actor and critic
        # params = list(self.actor.parameters()) + list(self.log_std)
        # self.actor_optim = Adam(itertools.chain(*params), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.log_std = torch.tensor(0.0, requires_grad=True)
        self.actor_optim.add_param_group({'params': self.log_std})

        # Initialize the covariance matrix used to query the actor for actions
        # self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        # self.cov_mat = torch.diag(self.cov_var)       

        # This logger will help us with printing out summaries of each iteration
        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,          # timesteps so far
            'i_so_far': 0,          # iterations so far
            'batch_lens': [],       # episodic lengths in batch
            'batch_rews': [],       # episodic returns in batch
            'actor_losses': [],     # losses of actor network in current iteration
            'critic_losses': [],
            'clipped_fraction': [],
            'batch_entropy': [],
        }

    def learn(self, total_timesteps):
        """
            Train the actor and critic networks. Here is where the main PPO algorithm resides.
            Parameters:
                total_timesteps - the total number of timesteps to train for
            Return:
                None
        """
        print(
            f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        t_so_far = 0  # Timesteps simulated so far
        i_so_far = 0  # Iterations ran so far
        # ALG STEP 2
        while t_so_far < total_timesteps:
            # decide if videos should be rendered/logged at this iteration
            if i_so_far % 10 == 0:  # video_log_freq, a hyperparameter
                self.log_video = True
            else:
                self.log_video = False

            batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_masks, batch_image_obs = self.rollout(
            )                     # ALG STEP 3

            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)

            # Increment the number of iterations
            i_so_far += 1

            # Logging timesteps so far and iterations so far
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            # Calculate advantage at k-th iteration
            V, _ = self.evaluate(batch_obs, batch_acts)
            # ALG STEP 5
            batch_returns = self.compute_returns(batch_rews, V.detach().numpy(), batch_masks)
            A_k = batch_returns - V.detach()

            # Normalizing advantages to decrease the variance and makes
            # convergence much more stable and faster.
            advantage = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            samples_get_clipped = 0
            # Update the network for some n epochs
            # ALG STEP 6 & 7
            for _ in range(self.n_updates_per_iteration):
                data_generator = self.generator(
                    batch_obs, batch_acts, batch_log_probs, batch_returns, advantage)
                
                for sample in data_generator:
                    obs_sample, actions_sample, log_probs_sample, returns_sample, advantages_sample = sample
                    
                    # Calculate V_phi and pi_theta(a_t | s_t)
                    V, curr_log_probs = self.evaluate(obs_sample, actions_sample)

                    # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
                    ratios = torch.exp(curr_log_probs - log_probs_sample)
                    
                    # Calculate surrogate losses.
                    surr1 = ratios * advantages_sample
                    surr2 = torch.clamp(ratios, 1 - self.clip, 1 +
                                        self.clip) * advantages_sample

                    # Calculate actor and critic losses.
                    # NOTE: take the negative min of the surrogate losses because we're trying to maximize
                    # the performance function, but Adam minimizes the loss. So minimizing the negative
                    # performance function maximizes it.
                    actor_loss = (-torch.min(surr1, surr2)).mean()
                    # clipped = torch.min(surr1, surr2)
                    # print("----", surr1.shape)   
                    bool_tensor = torch.all(torch.eq(surr1, surr2), dim=1).squeeze()
                    false_values = bool_tensor.masked_select(bool_tensor == False)
                    false_num = len(false_values)
                    samples_get_clipped += false_num
                    
                    # actor_loss = -clipped.mean()
                    self.actor_optim.zero_grad()
                    
                    critic_loss = nn.MSELoss()(V, returns_sample)
                    self.critic_optim.zero_grad()
                    
                    actor_loss.backward(retain_graph=True)
                    critic_loss.backward()                   
                    
                    self.actor_optim.step()  
                    self.critic_optim.step()                       

                    # Log actor and critic loss
                    self.logger['actor_losses'].append(actor_loss.detach())
                    self.logger['critic_losses'].append(critic_loss.detach())
            
            
            # 32 is the number of sampled batches, 64 is the mini_batch_size
            samples_get_clipped /= (self.n_updates_per_iteration * 32 * 64)
            self.logger['clipped_fraction'] = samples_get_clipped

            # Print a summary of our training so far
            self._log_summary()

            # log/save
            if self.log_video:
                # perform logging
                print('\nBeginning logging procedure...')
                # Need [N, T, C, H, W] input tensor for video logging
                video = torch.tensor(batch_image_obs).unsqueeze(0).unsqueeze(2)
                self.writer.add_video('{}'.format('train_rollouts'), video, i_so_far-1, fps=10)

    def generator(self, batch_obs, batch_acts, batch_log_probs, batch_returns, A_k):
        num_mini_batch = 32
        mini_batch_size = self.timesteps_per_batch // num_mini_batch # 2048/32
        batch_size = self.timesteps_per_batch
        
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:
            obs_sample = batch_obs[indices]
            actions_sample = batch_acts[indices]
            log_probs_sample = batch_log_probs[indices]
            returns_sample = batch_returns[indices]
            advantages_sample = A_k[indices]

            yield obs_sample, actions_sample, log_probs_sample, returns_sample, advantages_sample

    def rollout(self):
        """
            Collect the batch of data from simulation. Since this is an on-policy algorithm, a fresh batch of data need to 
            be collected each time as we iterate the actor/critic networks.
            Parameters:
                None
            Return:
                batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
                batch_acts - the actions collected this batch. Shape: (number of timesteps, dimension of action)
                batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
                batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
                batch_lens - the lengths of each episode this batch. Shape: (number of episodes)
        """
        # Batch data
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []
        batch_masks = []
        batch_image_obs = []
        batch_entropy = []

        # Episodic data. Keeps track of rewards per episode, will get cleared
        # upon each new episode
        ep_rews = []

        t = 0  # Keeps track of how many timesteps we've run so far this batch

        # Keep simulating until we've run more than or equal to specified timesteps per batch
        while t < self.timesteps_per_batch:
            # ep_rews = []  # rewards collected per episode

            # Reset the environment.
            obs = self.env.reset()
            done = False

            # Run an episode for a maximum of max_timesteps_per_episode timesteps
            for ep_t in range(self.max_timesteps_per_episode):
                # Prepare video
                batch_image_obs.append(self.env.render())

                t += 1  # Increment timesteps ran this batch so far

                # Track observations in this batch
                # if self.encoder:
                    # obs = self.encoder(obs[None, None, :]).detach().numpy()
                    # obs = self.encoder(obs[None, None, :], detach=True).detach().numpy()        
                # obs = torch.from_numpy(obs[None, :]).float()
                batch_obs.append(obs)

                # Calculate action and make a step in the env.
                action, log_prob, entropy = self.get_action(obs)
                obs, rew, done, _ = self.env.step(action)

                # Track recent reward, action, and action log probability
                # ep_rews.append(rew)
                batch_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                batch_entropy.append(entropy)

                # If the environment tells us the episode is terminated, break
                if done:
                    batch_masks.append(0.0)
                    break
                else:
                    batch_masks.append(1.0)

            # Track episodic lengths and rewards
            batch_lens.append(ep_t + 1)
            # batch_rews.append(ep_rews)

        # Reshape data as tensors in the shape specified in function description, before returning
        # if isinstance(batch_obs[0], torch.Tensor):
        # 	batch_obs = torch.stack(batch_obs, dim=0)       
        # else:
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        # ALG STEP 4
        # batch_rtgs = self.compute_rtgs(batch_rews)

        # Log the episodic returns and episodic lengths in this batch.
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens
        self.logger['batch_entropy'] = batch_entropy

        return batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_masks, batch_image_obs

    def compute_rtgs(self, batch_rews):
        """
            Compute the Reward-To-Go of each timestep in a batch given the rewards.
            Parameters:
                batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)
            Return:
                batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
        """
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []

        # Iterate through each episode
        for ep_rews in reversed(batch_rews):

            discounted_reward = 0  # The discounted reward so far

            # Iterate through all rewards in the episode.
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs
    
    def compute_returns(self, batch_rews, V, batch_masks):
        """
            Compute the GAE of each timestep in a batch given the rewards.
            Parameters:
                batch_rews - the rewards in a batch, Shape: (number of timesteps in batch)
                V - the reward prediction, Shape: (number of timesteps in batch)
            Return:
                batch_returns - Shape: (number of timesteps in batch)
        """
        batch_returns = []
        batch_masks.append(1.0)
        V = np.append(V, V[-1])
        gae = 0
        for step in reversed(range(len(batch_rews))):
            delta = batch_rews[step] + self.gamma * V[step + 1] * batch_masks[step + 1] - V[step]
            gae = delta + self.gamma * self.gae_lambda * batch_masks[step + 1] * gae
            batch_returns.insert(0, gae + V[step])

        # Convert the batch_returns into a tensor
        batch_returns = torch.tensor(batch_returns, dtype=torch.float)

        return batch_returns

    def get_action(self, obs):
        """
            Queries an action from the actor network, should be called from rollout.
            Parameters:
                obs - the observation at the current timestep
            Return:
                action - the action to take, as a numpy array
                log_prob - the log probability of the selected action in the distribution
        """
        # Query the actor network for a mean action
        mean = self.actor(obs)
        
        # Create a distribution with the mean action and std from the covariance matrix above.
        dist = Normal(mean, torch.exp(self.log_std))

        # Sample an action from the distribution
        action = dist.sample()

        # Calculate the log probability for that action
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        # Record the entropy of policy
        entropy = dist.entropy().mean()

        # Return the sampled action and the log probability of that action in our distribution
        return action.detach().numpy(), log_prob.detach().numpy(), entropy.detach().numpy()

    def evaluate(self, batch_obs, batch_acts):
        """
            Estimate the values of each observation, and the log probs of
            each action in the most recent batch with the most recent
            iteration of the actor network. Should be called from learn.
            Parameters:
                batch_obs - the observations from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of observation)
                batch_acts - the actions from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of action)
            Return:
                V - the predicted values of batch_obs
                log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
        """
        # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        V = self.critic(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using most recent actor network.
        mean = self.actor(batch_obs)
        dist = Normal(mean, torch.exp(self.log_std))
        log_probs = dist.log_prob(batch_acts).sum(-1, keepdim=True)

        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return V, log_probs

    def _init_hyperparameters(self, hyperparameters):
        """
            Initialize default and custom values for hyperparameters
            Parameters:
                hyperparameters - the extra arguments included when creating the PPO model, should only include
                                    hyperparameters defined below with custom values.
            Return:
                None
        """
        # Initialize default values for hyperparameters
        # Algorithm hyperparameters
        # Number of timesteps to run per batch
        self.timesteps_per_batch = 4800
        # Max number of timesteps per episode
        self.max_timesteps_per_episode = 1600
        # Number of times to update actor/critic per iteration
        self.n_updates_per_iteration = 5
        self.lr = 0.005                                 # Learning rate of actor optimizer
        # Discount factor to be applied when calculating Rewards-To-Go
        self.gamma = 0.95
        self.gae_lambda = 0.95
        # Recommended 0.2, helps define the threshold to clip the ratio during SGA
        self.clip = 0.2

        # Miscellaneous parameters
        # If we should render during rollout
        self.render = True
        self.render_every_i = 10                        # Only render every n iterations
        # How often we save in number of iterations
        self.save_freq = 10
        # Sets the seed of our program, used for reproducibility of results
        self.seed = None

        # Change any default values to custom values for specified hyperparameters
        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))

        # Sets the seed if specified
        if self.seed != None:
            # Check if our seed is valid first
            assert(type(self.seed) == int)

            # Set the seed
            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")

    def _log_summary(self):
        """
            Print to stdout what we've logged so far in the most recent batch.
            Parameters:
                None
            Return:
                None
        """
        # Calculate logging values.
        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        clipped_fraction = self.logger['clipped_fraction']
        # avg_ep_rews = np.mean([np.sum(rew)
        #                       for rew in self.logger['batch_rews']])
        avg_ep_rews = np.sum(self.logger['batch_rews']) / len(self.logger['batch_lens'])
        avg_actor_loss = np.mean([losses.float().mean()
                                 for losses in self.logger['actor_losses']])
        avg_critic_loss = np.mean([losses.float().mean()
                                  for losses in self.logger['critic_losses']])
        avg_entropy = np.mean(self.logger['batch_entropy'])
        
        # print("---", avg_ep_rews)
        self.writer.add_scalar('Average Episodic Return', avg_ep_rews, i_so_far)
        self.writer.add_scalar('Average Actor Loss', avg_actor_loss, i_so_far)
        self.writer.add_scalar('Average Critic Loss', avg_critic_loss, i_so_far)
        self.writer.add_scalar('Average Policy Entropy', avg_entropy, i_so_far)
        self.writer.add_scalar('Clipped portion', clipped_fraction, i_so_far)

        if i_so_far % 10 == 0:
            # Round decimal places for more aesthetic logging messages
            avg_ep_rews = str(round(avg_ep_rews, 2))
            avg_actor_loss = str(round(avg_actor_loss, 5))
            clipped_fraction = str(round(clipped_fraction, 5))
            avg_entropy = str(round(avg_entropy, 5))

            # Print logging statements
            print(flush=True)
            print(
                f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
            print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
            print(f"Average Loss: {avg_actor_loss}", flush=True)
            print(f"Average Entropy: {avg_entropy}", flush=True)
            print(f"Clipped Fraction: {clipped_fraction}", flush=True)
            print(f"Timesteps So Far: {t_so_far}", flush=True)
            print(f"------------------------------------------------------", flush=True)
            print(flush=True)

        # Reset batch-specific logging data
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []
        self.logger['batch_lens'] = []
        self.logger['clipped_fraction'] = []
        self.logger['batch_entropy'] = []


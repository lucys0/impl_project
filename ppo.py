import gym
import time

import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal

"""
    This PPO implementation follows the pseudocode provided in OpenAI's Spinning Up for PPO: 
    https://spinningup.openai.com/en/latest/algorithms/ppo.html. 
    Pseudocode line numbers are specified as "ALG STEP #" in ppo.py.
"""

class PPO:
	def __init__(self, policy_class, env, **hyperparameters):
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
		self.obs_dim = env.observation_space.shape[0]
		self.act_dim = env.action_space.shape[0]

	 # Initialize actor and critic networks
		# ALG STEP 1
		self.actor = policy_class(self.obs_dim, self.act_dim)
		self.critic = policy_class(self.obs_dim, 1)

		# Initialize optimizers for actor and critic
		self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
		self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

		# Initialize the covariance matrix used to query the actor for actions
		self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
		self.cov_mat = torch.diag(self.cov_var)

		# This logger will help us with printing out summaries of each iteration
		self.logger = {
			'delta_t': time.time_ns(),
			't_so_far': 0,          # timesteps so far
			'i_so_far': 0,          # iterations so far
			'batch_lens': [],       # episodic lengths in batch
			'batch_rews': [],       # episodic returns in batch
			'actor_losses': [],     # losses of actor network in current iteration
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
			batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout(
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
			A_k = batch_rtgs - V.detach()

			# One of the only tricks I use that isn't in the pseudocode. Normalizing advantages
			# isn't theoretically necessary, but in practice it decreases the variance of
			# our advantages and makes convergence much more stable and faster. I added this because
			# solving some environments was too unstable without it.
			A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

			# This is the loop where we update our network for some n epochs
			# ALG STEP 6 & 7
			for _ in range(self.n_updates_per_iteration):
				# Calculate V_phi and pi_theta(a_t | s_t)
				V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

				# Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
				ratios = torch.exp(curr_log_probs - batch_log_probs)

				# Calculate surrogate losses.
				surr1 = ratios * A_k
				surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

				# Calculate actor and critic losses.
				# NOTE: take the negative min of the surrogate losses because we're trying to maximize
				# the performance function, but Adam minimizes the loss. So minimizing the negative
				# performance function maximizes it.
				actor_loss = (-torch.min(surr1, surr2)).mean()
				critic_loss = nn.MSELoss()(V, batch_rtgs)

				# Calculate gradients and perform backward propagation for actor network
				self.actor_optim.zero_grad()
				actor_loss.backward(retain_graph=True)
				self.actor_optim.step()

				# Calculate gradients and perform backward propagation for critic network
				self.critic_optim.zero_grad()
				critic_loss.backward()
				self.critic_optim.step()

				# Log actor loss
				self.logger['actor_losses'].append(actor_loss.detach())

			# Print a summary of our training so far
			self._log_summary()

			# Save our model if it's time | needed?
			if i_so_far % self.save_freq == 0:
				torch.save(self.actor.state_dict(), './ppo_actor.pth')
				torch.save(self.critic.state_dict(), './ppo_critic.pth')

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

		# Episodic data. Keeps track of rewards per episode, will get cleared
		# upon each new episode
		ep_rews = []

		t = 0  # Keeps track of how many timesteps we've run so far this batch

		# Keep simulating until we've run more than or equal to specified timesteps per batch
		while t < self.timesteps_per_batch:
			ep_rews = []  # rewards collected per episode

			# Reset the environment.
			obs = self.env.reset()
			done = False

			# Run an episode for a maximum of max_timesteps_per_episode timesteps
			for ep_t in range(self.max_timesteps_per_episode):
				# If render is specified, render the environment
				if self.render and (self.logger['i_so_far'] % self.render_every_i == 0) and len(batch_lens) == 0:
					self.env.render()

				t += 1  # Increment timesteps ran this batch so far

				# Track observations in this batch
				batch_obs.append(obs)

				# Calculate action and make a step in the env.
				# Note that rew is short for reward.
				action, log_prob = self.get_action(obs)
				obs, rew, done, _ = self.env.step(action)

				# Track recent reward, action, and action log probability
				ep_rews.append(rew)
				batch_acts.append(action)
				batch_log_probs.append(log_prob)

				# If the environment tells us the episode is terminated, break
				if done:
					break

			# Track episodic lengths and rewards
			batch_lens.append(ep_t + 1)
			batch_rews.append(ep_rews)

		# Reshape data as tensors in the shape specified in function description, before returning
		batch_obs = torch.tensor(batch_obs, dtype=torch.float)
		batch_acts = torch.tensor(batch_acts, dtype=torch.float)
		batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
		# ALG STEP 4
		batch_rtgs = self.compute_rtgs(batch_rews)

		# Log the episodic returns and episodic lengths in this batch.
		self.logger['batch_rews'] = batch_rews
		self.logger['batch_lens'] = batch_lens

		return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

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
		dist = MultivariateNormal(mean, self.cov_mat)

		# Sample an action from the distribution
		action = dist.sample()

		# Calculate the log probability for that action
		log_prob = dist.log_prob(action)

		# Return the sampled action and the log probability of that action in our distribution
		return action.detach().numpy(), log_prob.detach()

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
		dist = MultivariateNormal(mean, self.cov_mat)
		log_probs = dist.log_prob(batch_acts)

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
		# Calculate logging values. I use a few python shortcuts to calculate each value
		# without explaining since it's not too important to PPO; feel free to look it over,
		# and if you have any questions you can email me (look at bottom of README)
		delta_t = self.logger['delta_t']
		self.logger['delta_t'] = time.time_ns()
		delta_t = (self.logger['delta_t'] - delta_t) / 1e9
		delta_t = str(round(delta_t, 2))

		t_so_far = self.logger['t_so_far']
		i_so_far = self.logger['i_so_far']
		avg_ep_lens = np.mean(self.logger['batch_lens'])
		avg_ep_rews = np.mean([np.sum(ep_rews)
		                      for ep_rews in self.logger['batch_rews']])
		avg_actor_loss = np.mean([losses.float().mean()
		                         for losses in self.logger['actor_losses']])

		# Round decimal places for more aesthetic logging messages
		avg_ep_lens = str(round(avg_ep_lens, 2))
		avg_ep_rews = str(round(avg_ep_rews, 2))
		avg_actor_loss = str(round(avg_actor_loss, 5))

		# Print logging statements
		print(flush=True)
		print(
			f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
		print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
		print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
		print(f"Average Loss: {avg_actor_loss}", flush=True)
		print(f"Timesteps So Far: {t_so_far}", flush=True)
		print(f"Iteration took: {delta_t} secs", flush=True)
		print(f"------------------------------------------------------", flush=True)
		print(flush=True)

		# Reset batch-specific logging data
		self.logger['batch_lens'] = []
		self.logger['batch_rews'] = []
		self.logger['actor_losses'] = []

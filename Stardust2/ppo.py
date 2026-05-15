from utils import *


class PolicyNetwork(nn.Module):
    def __init__(self, observation_space, action_space):
        super(PolicyNetwork, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        num_nodes, input_dim = observation_space.shape
        self.num_nodes = num_nodes
        self.input_dim = input_dim

        # Define the common network
        self.norm=torch.nn.LayerNorm(input_dim)
        self.norm1=torch.nn.LayerNorm(256)
        self.common_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
        )

        # Define the actor network
        # For MultiDiscrete action space, output logits for each action component
        # action_space.nvec = [num_nodes, grid_size]
        self.actor_node = nn.Linear(256, 1)  # logits for node selection
        self.actor_position = nn.Sequential(nn.Linear(256, self.action_space.nvec[1]))
                                            #nn.ReLU(),
                                            #nn.Linear(self.action_space.nvec[1],1))   # logits for position selection

        # Define the critic network
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        #pdb.set_trace()
        x = x.float()
        x = self.norm(x)
        x = self.common_net(x)
        x=self.norm1(x)
       # Actor outputs
        logits_node = self.actor_node(x).squeeze(-1)
        #xf = x.mean(dim=0)
        logits_position = self.actor_position(x)
        logits_position = torch.mean(logits_position,dim=1)
       # Critic output
        value = self.critic(x)
        return logits_node,logits_position, value

# Define the PPO Agent
class PPOAgent:
    def __init__(self, env, policy_network, optimizer, clip_param=0.2, max_grad_norm=0.5, ppo_epochs=10, batch_size=64, gamma=0.99, lam=0.95):
        self.env = env
        self.policy = policy_network
        self.optimizer = optimizer
        self.clip_param = clip_param
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.lam = lam

        # Initialize storage for rollouts
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def select_action(self, observation):
        # Get action logits and value from the policy network
        logits_node, logits_position, value = self.policy(observation.unsqueeze(0))  # Add batch dimension

        # Sample actions
        # For MultiDiscrete action space, sample each component separately

        # For node selection
        dist_node = torch.distributions.Categorical(logits=logits_node)
        action_node = dist_node.sample()
        log_prob_node = dist_node.log_prob(action_node)

        # For position selection
        dist_position = torch.distributions.Categorical(logits=logits_position)
        action_position = dist_position.sample()
        log_prob_position = dist_position.log_prob(action_position)

        # Combine actions and log_probs
        action = torch.stack([action_node.squeeze(), action_position.squeeze()], dim=-1)
        log_prob = log_prob_node + log_prob_position

        return action, log_prob.squeeze(), value.squeeze()

    def store_transition(self, observation, action, log_prob, reward, value, done):
        self.observations.append(observation)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(torch.tensor([reward], dtype=torch.float32))
        self.values.append(value)
        self.dones.append(torch.tensor([done], dtype=torch.float32))

    def compute_gae(self, next_value):
        # Compute generalized advantage estimation
        values = self.values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + self.gamma * values[step + 1] * (1 - self.dones[step]) - values[step]
            gae = delta + self.gamma * self.lam * (1 - self.dones[step]) * gae
            returns.insert(0, gae + values[step])
        self.returns = returns

    def update(self):
        # Convert lists to tensors
        observations = torch.stack(self.observations).detach()
        actions = torch.stack(self.actions).detach()
        log_probs_old = torch.stack(self.log_probs).detach()
        returns = torch.stack(self.returns).detach()
        values = torch.stack(self.values).detach()

        # Normalize advantages
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        num_steps = observations.shape[0]

        # For PPO update, shuffle data and create mini-batches
        indices = np.arange(num_steps)
        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, num_steps, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_obs = observations[batch_indices]
                batch_actions = actions[batch_indices]
                batch_log_probs_old = log_probs_old[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                # Get current policy outputs
                logits_node, logits_position, values = self.policy(batch_obs)
                dist_node = torch.distributions.Categorical(logits=logits_node)
                dist_position = torch.distributions.Categorical(logits=logits_position)

                # Get log probs for actions
                log_prob_node = dist_node.log_prob(batch_actions[:, 0])
                log_prob_position = dist_position.log_prob(batch_actions[:, 1])
                log_prob = log_prob_node + log_prob_position

                # Compute ratio
                ratio = torch.exp(log_prob - batch_log_probs_old)

                #pdb.set_trace()
                # Compute surrogate loss
                surr1 = ratio.unsqueeze(1) * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param).unsqueeze(1) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                #pdb.set_trace()
                # Value function loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)

                # Total loss
                loss = policy_loss + 0.5 * value_loss
                #return loss
                # Optimize
                #pdb.set_trace()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.zero_grad()
                self.optimizer.step()

    def train(self, total_timesteps):
        observation, _,_ = self.env.reset(True)
        print(self.env.current_graph_name)
        observation = observation.to(next(self.policy.parameters()).device)
        episode_rewards = []
        episode_reward = 0
        timestep = 0
        while timestep < total_timesteps:
            action, log_prob, value = self.select_action(observation)
            action_np = action.cpu().numpy()
            next_observation, reward, done, _, _ = self.env.step(action_np,True)
            next_observation = next_observation.to(next(self.policy.parameters()).device)
            self.store_transition(observation, action, log_prob, reward, value, done)

            observation = next_observation
            episode_reward += reward
            timestep += 1

            if done:
                next_value = torch.tensor([0.0], dtype=torch.float32).to(next(self.policy.parameters()).device)
                self.compute_gae(next_value)
                self.update()
                #loss.backward()
                #nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                #self.optimizer.zero_grad()
                #self.optimizer.step()
                # Clear storage
                self.observations = []
                self.actions = []
                self.log_probs = []
                self.rewards = []
                self.values = []
                self.dones = []
                #pdb.set_trace()
                # Record episode reward
                episode_rewards.append(episode_reward)
                print(f"Episode reward: {episode_reward}")
                episode_reward = 0
                print(self.env.current_graph_name)
                # Reset environment
                observation, _,_ = self.env.reset(False)
                observation = observation.to(next(self.policy.parameters()).device)
        return episode_rewards
        #print(f"Training completed over {total_timesteps} timesteps")
        #print(f"Average episode reward: {np.mean(episode_rewards)}")
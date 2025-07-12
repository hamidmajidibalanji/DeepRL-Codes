# CS-545 Implementation for RevealMNIST Project.
# DDQN Implementation
# VErsion: Determinstic POlicy
# In stochastic version
# 1. More training episodes to reach a stable learning.
# 2. Exploration strategy: decay epsilon more slowly to ensure sufficient exploration
# due to more randomness in the environment.   ---> Higher epsilon_min = 0.1
# 3. Larger Batch size to smooth out the noise in gradient estimates.




# Import Modules
import random as random
import numpy as np  
from collections import deque   
import torch
import torch.nn as nn  
import torch.optim as optim   
import gymnasium as gym
import reveal_mnist
import matplotlib.pyplot as plt



import time  

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)



# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



NUM_EPISODES = 10000

# Q-Network
class QNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, action_dim)
        )

    def forward(self, x):
        return self.net(x)




class DoubleDQNAgent:
    def __init__(self, obs_shape, action_space, gamma=0.90, lr=5e-5, batch_size=128, buffer_size=300000):
        self.obs_dim = int(np.prod(obs_shape))
        self.n_actions = action_space.n  
        self.gamma = gamma
        self.batch_size = batch_size
        

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = QNetwork(self.obs_dim, self.n_actions).to(self.device)
        self.target_net = QNetwork(self.obs_dim, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr, weight_decay=1e-5)
        self.memory = deque(maxlen=buffer_size)

        self.epsilon = 1.0
        self.epsilon_min = 0.2
        self.epsilon_decay = 0.995  # First: 0.995 second:0.99(Bad results)

        # Target network update tracking
        self.learn_step_counter = 0
        self.target_update_freq = 10  # Update target net every 1000 steps

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions-1)
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state)
        return q_values.argmax().item()

    def step(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # Load Replay Buffer
        if len(self.memory) >= self.batch_size:
            self.learn()

    def learn(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)  # Sample from the Buffer

        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.q_net(states).gather(1, actions)

        # Double DQN target computation
        next_actions = self.q_net(next_states).argmax(1).unsqueeze(1)
        next_q_values = self.target_net(next_states).gather(1, next_actions)
        targets = rewards + (1 - dones) * self.gamma * next_q_values  

        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()  

        # Epsilon decay
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        # Update target network every fixed number of steps
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())




# Create the environment with appropriate device
env = gym.make(
    'RevealMNIST-v0',
    classifier_model_weights_loc="/home/hamid/RL_Final_Project/RevealMNIST/mnist_predictor_masked.pt",
    device=device.type,  # 'cuda' or 'cpu'
    visualize=False,
    stochastic=True
)


agent = DoubleDQNAgent(env.observation_space.shape, env.action_space)




# Metrics
accuracies = []
rewards_per_episode = []
reveal_efficiencies = []
epsilons = []
num_steps = []
training_time = []



# Training
start_time = time.time()

for ep in range(NUM_EPISODES):
    state, _ = env.reset()
    done = False
    cumulative_reward = 0
    correctly_pred = 0
    reveal_percent = 0
    length=0

    while not done:
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        agent.step(state, action, reward, next_state, done)
        total_time = time.time() - start_time
        #print(f"Training took {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
        training_time.append(total_time)


        state = next_state
        cumulative_reward += reward

        length += 1


        if done:
            final_reward = reward

        # Uncomment to visualize the environment
        #env.render()
    if action == 4:
        if reward >= 0:
            reward += 5  # small bonus for correct
            if state[-1] < 0.2:
                reward += 5  # reward efficient correct decision
        else:
            reward -= 20  # strong penalty for wrong guesses



    if action == 4 and reward >= 0 and terminated:
        correctly_pred = 1
        reveal_percent = state[-1]
    elif action == 4 and reward < 0 and terminated:
        correctly_pred = 0
        reveal_percent = state[-1]  

    print(f"Episode {ep+1} - Total Reward: {cumulative_reward:.2f} ")


    accuracies.append(correctly_pred)
    rewards_per_episode.append(cumulative_reward)
    reveal_efficiencies.append(reveal_percent)
    epsilons.append(agent.epsilon)
    num_steps.append(length)


# Summary statistics
accuracy = np.mean(accuracies)
avg_reward = np.mean(rewards_per_episode)
avg_reveal_efficiency = np.mean(reveal_efficiencies)
avg_time_steps = np.mean(num_steps)
avg_epsilon = np.mean(epsilons)
avg_training_time = np.mean(training_time)


# Q-Network
torch.save(agent.q_net.state_dict(), "ddqn_trained_model_stochastic.pt")

env.close()



print("\n=== Evaluation Summary ===")
print(f"Classification Accuracy: {accuracy * 100:.2f}%")
print(f"Average Reward per Episode: {avg_reward:.2f}")
print(f"Average Reveal Efficiency: {avg_reveal_efficiency * 100:.2f}%")
print(f"Average Time Steps: {avg_time_steps :.2f}")
print(f"Average Epsilon: {avg_epsilon:.2f}")
print(f"Average Training Time: {avg_training_time:.2f}")



print(f"Length accuracies: {len(accuracies)}")
print(f"Length rewards_per_episode: {len(rewards_per_episode)}")
print(f"Length reveal_efficiencies: {len(reveal_efficiencies)}")
print(f"Length Time Steps:{len(num_steps)}")
print(f"Lenghth Epsilon: {len(epsilons)}")
print(f"Training Time: {len(training_time)}")

# Save the data
with open("accur_ddqn.txt", "w") as f:
    for item in accuracies:
        f.write(f"{item}\n")

with open("reward_ddqn.txt", "w") as f:
    for item in rewards_per_episode:
        f.write(f"{item}\n")

with open("reveal_ddqn.txt", "w") as f:
    for item in reveal_efficiencies:
        f.write(f"{item}\n")


with open("epsilon_ddqn.txt", "w") as f:
    for item in epsilons:
        f.write(f"{item}\n")


with open("num_time_step_ddqn.txt", "w") as f:
	for item in num_steps:
		f.write(f"{item}\n") 



with open("training_time_ddqn.txt", "w") as f:
	for item in training_time:
		f.write(f"{item}\n")





# Plotting

# Rewards
plt.figure()
plt.plot(rewards_per_episode)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('DDQN-RevealMNIST Rewards vs. Episodes')
plt.savefig("rewards_plot_ddqn.png")


# Accuracy
plt.figure()
plt.plot(accuracies)
plt.xlabel('Episode')
plt.ylabel('Accuracy')
plt.title('DDQN-RevealMNIST Accuracy vs. Episodes')
plt.savefig("accuracy_plot_ddqn")


# Reveal EFficiency
plt.figure()
plt.plot(reveal_efficiencies)
plt.xlabel('Episode')
plt.ylabel('Reveal Efficiency')
plt.title('DDQN-RevealMNIST Reveal Efficiency vs. Episodes')
plt.savefig("revealEfficiency_plot_ddqn")


# Epsilon
plt.figure()
plt.plot(epsilons)
plt.xlabel('Episode')
plt.ylabel('Epsilon Decay')
plt.title('DDQN-RevealMNIST Epsilon vs. Episodes')
plt.savefig("epsilon_decay_plot_ddqn")


# Time steps
plt.figure()
plt.plot(num_steps)
plt.xlabel('Episode')
plt.ylabel('Time Steps')
plt.title('DDQN-RevealMNIST Time Steps vs. Episodes')
plt.savefig("Timesteps_plot_ddqn")


# Training Time
plt.figure()
plt.plot(training_time)
plt.xlabel('Episode')
plt.ylabel('Training Time')
plt.title('DDQN-RevealMNIST Training Time vs. Episodes')
plt.savefig("training_time_plot_ddqn")



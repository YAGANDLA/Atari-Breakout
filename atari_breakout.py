import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
import os
import logging
import imageio
import time

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Hyperparameters
EPISODES = 1500
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.02  # Slightly lower for better exploitation
EXPLORATION_STEPS = 250000  # Faster decay
LEARNING_RATE = 0.0001  # Reduced for stability
MEMORY_SIZE = 100000
TARGET_UPDATE = 1000  # More frequent updates
TRAIN_START = 10000  # Start earlier
NO_OP_STEPS = 30
VIDEO_INTERVAL = 500
DEFAULT_RENDER_FPS = 30
VIDEO_EPISODES = 2
MAX_VIDEO_FRAMES = 900

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        conv_h = ((input_shape[1] - 8) // 4 + 1 - 4) // 2 + 1 - 2
        conv_w = ((input_shape[2] - 8) // 4 + 1 - 4) // 2 + 1 - 2
        self.fc1 = nn.Linear(conv_w * conv_h * 64, 512)
        self.fc2 = nn.Linear(512, num_actions)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def huber_loss(y_pred, y_true):
    error = y_pred - y_true
    abs_error = torch.abs(error)
    quadratic = torch.min(abs_error, torch.tensor(1.0))
    linear = abs_error - quadratic
    return torch.mean(0.5 * quadratic * quadratic + linear)

def preprocess_frame(frame):
    frame = Image.fromarray(frame).convert('L').resize((84, 84))
    return T.ToTensor()(frame)

def save_video(env, policy_net, device, episode, filename="breakout_episode_{}.mp4"):
    frames = []
    for ep in range(VIDEO_EPISODES):
        if len(frames) >= MAX_VIDEO_FRAMES:
            break
        observation = env.reset()[0]
        state = preprocess_frame(observation)
        history = torch.cat([state] * 4, dim=0).unsqueeze(0).to(device)
        done = False
        
        for _ in range(NO_OP_STEPS):
            if len(frames) >= MAX_VIDEO_FRAMES:
                break
            observation, _, _, _, _ = env.step(1)
            frames.append(env.render())
        
        while not done and len(frames) < MAX_VIDEO_FRAMES:
            with torch.no_grad():
                action = policy_net(history).max(1)[1].item()
            real_action = [1, 2, 3][action]
            next_observation, _, terminated, truncated, _ = env.step(real_action)
            done = terminated or truncated
            frames.append(env.render())
            next_state = preprocess_frame(next_observation).unsqueeze(0)
            history = torch.cat([next_state, history[:, 1:, :, :]], dim=1)
    
    with imageio.get_writer(filename.format(episode), fps=DEFAULT_RENDER_FPS, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame)
    logger.info(f"Video saved as {filename.format(episode)} with {len(frames)} frames")

def train_dqn():
    env = gym.make('BreakoutDeterministic-v4', render_mode='rgb_array')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    n_actions = 3
    policy_net = DQN((4, 84, 84), n_actions).to(device)
    target_net = DQN((4, 84, 84), n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.RMSprop(policy_net.parameters(), lr=LEARNING_RATE, eps=0.01)
    memory = deque(maxlen=MEMORY_SIZE)
    
    epsilon = EPSILON_START
    global_step = 0
    episode_rewards = []
    losses = []
    
    for episode in range(EPISODES):
        observation = env.reset()[0]
        state = preprocess_frame(observation)
        history = torch.cat([state] * 4, dim=0).unsqueeze(0).to(device)
        
        for _ in range(random.randint(1, NO_OP_STEPS)):
            observation, _, _, _, _ = env.step(1)
        
        total_reward = 0
        done = False
        episode_loss = 0
        steps = 0
        start_life = 5
        
        while not done:
            global_step += 1
            steps += 1
            
            action = random.randrange(n_actions) if random.random() < epsilon else policy_net(history).max(1)[1].item()
            real_action = [1, 2, 3][action]
            next_observation, reward, terminated, truncated, info = env.step(real_action)
            done = terminated or truncated
            reward = np.clip(reward, -1., 1.)
            
            next_state = preprocess_frame(next_observation).unsqueeze(0)
            next_history = torch.cat([next_state, history[:, 1:, :, :]], dim=1).to(device)
            
            dead = start_life > info['lives']
            if dead:
                start_life = info['lives']
            
            memory.append((history, action, reward, next_history, dead))
            history = next_history if not dead else torch.cat([next_state] * 4, dim=1).to(device)
            
            total_reward += reward
            
            if len(memory) >= TRAIN_START:  # Train every step
                if global_step % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())
                
                batch = random.sample(memory, BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)
                
                states = torch.cat(states).to(device)
                actions = torch.tensor(actions, device=device)
                rewards = torch.tensor(rewards, device=device)
                next_states = torch.cat(next_states).to(device)
                dones = torch.tensor(dones, dtype=torch.bool, device=device)
                
                q_values = policy_net(states).gather(1, actions.unsqueeze(1))
                with torch.no_grad():
                    next_q_values = target_net(next_states).max(1)[0]
                    targets = rewards + (GAMMA * next_q_values * (~dones))
                
                loss = huber_loss(q_values.squeeze(), targets)
                episode_loss += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
                optimizer.step()
            
            if done:
                avg_loss = episode_loss / steps if len(memory) >= TRAIN_START else 0
                episode_rewards.append(total_reward)
                losses.append(avg_loss)
                logger.info(f"Episode {episode}, Reward: {total_reward}, Steps: {steps}, "
                            f"Epsilon: {epsilon:.3f}, Avg Loss: {avg_loss:.4f}, Memory: {len(memory)}")
        
        epsilon = max(EPSILON_END, EPSILON_START - (EPSILON_START - EPSILON_END) * (global_step / EXPLORATION_STEPS))
        
        if episode % VIDEO_INTERVAL == 0:
            save_path = f"breakout_dqn_{episode}.pth"
            torch.save(policy_net.state_dict(), save_path)
            logger.info(f"Model saved at {save_path}")
            save_video(env, policy_net, device, episode)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, label='Reward')
    plt.title('Episode Rewards (1500 Episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(losses, label='Loss', color='orange')
    plt.title('Training Loss (1500 Episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.show()
    
    save_video(env, policy_net, device, "final")
    env.close()

if __name__ == "__main__":
    try:
        train_dqn()
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
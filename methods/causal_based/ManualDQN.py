import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Save a transition"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Randomly sample a batch of transitions"""
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)
    


class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, buffer_size=10000, device="cpu"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.device = device

        # Initialize Policy Network (local) and Target Network
        # num_outputs=1 because each 'treatment' head outputs a single Q-value
        self.policy_net = TARNet(num_covariates=state_dim, num_treatments=action_dim, num_outputs=1).to(device)
        self.target_net = TARNet(num_covariates=state_dim, num_treatments=action_dim, num_outputs=1).to(device)
        
        # Copy weights from policy to target
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target net is only used for inference

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        self.loss_fn = nn.MSELoss()

    def get_q_values(self, model, state_tensor):
        """
        Helper to handle TARNet's specific output format.
        TARNet returns a list of [Batch, 1] tensors. 
        We stack them to get [Batch, Action_Dim].
        """
        # outs is a list of tensors of shape (batch, 1)
        outs = model(state_tensor) 
        
        # Concatenate along dim 1 to get (batch, num_actions)
        q_values = torch.cat(outs, dim=1)
        return q_values

    def act(self, state, epsilon=0.1):
        """Epsilon-greedy action selection"""
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.get_q_values(self.policy_net, state)
            action = q_values.argmax(dim=1).item()
            
        return action

    def update(self, batch_size):
        if len(self.memory) < batch_size:
            return None # Not enough samples yet

        # 1. Sample Replay Buffer
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        # Convert to tensors
        states      = torch.FloatTensor(states).to(self.device)
        actions     = torch.LongTensor(actions).unsqueeze(1).to(self.device) # Shape: (batch, 1)
        rewards     = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # 2. Compute Q(s, a) using Policy Net
        # Get Q-values for all actions, then gather the one actually taken
        curr_q_values = self.get_q_values(self.policy_net, states)
        q_val = curr_q_values.gather(1, actions)

        # 3. Compute Target Q(s', a') using Target Net
        with torch.no_grad():
            next_q_values = self.get_q_values(self.target_net, next_states)
            max_next_q = next_q_values.max(dim=1, keepdim=True)[0]
            
            # Bellman Equation
            expected_q_val = rewards + (self.gamma * max_next_q * (1 - dones))

        # 4. Compute Loss and Optimize
        loss = self.loss_fn(q_val, expected_q_val)

        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Hard update: Copy weights from policy to target"""
        self.target_net.load_state_dict(self.policy_net.state_dict())


if __name__ == "__main__":
    # --- Configuration ---
    STATE_DIM = 10   # Example: 10 input features
    ACTION_DIM = 4   # Example: 4 possible actions
    BATCH_SIZE = 32
    
    # Initialize Agent
    agent = DQNAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM, device="cpu")

    # --- Dummy Training Loop ---
    print("Starting Dummy Training Loop...")
    
    # Simulate 100 steps
    fake_state = np.random.randn(STATE_DIM)
    
    for step in range(100):
        # 1. Select Action
        action = agent.act(fake_state, epsilon=0.1)

        # 2. Simulate Environment Step (Mock data)
        next_state = np.random.randn(STATE_DIM)
        reward = np.random.random() # Random reward
        done = (step % 20 == 0) # Periodically 'finish' episode

        # 3. Store in Buffer
        agent.memory.push(fake_state, action, reward, next_state, done)
        
        # 4. Update Network
        loss = agent.update(BATCH_SIZE)

        if done:
            fake_state = np.random.randn(STATE_DIM)
        else:
            fake_state = next_state

        if step % 10 == 0 and loss is not None:
            print(f"Step: {step}, Action: {action}, Loss: {loss:.4f}")

        # 5. Periodically update target network
        if step % 50 == 0:
            agent.update_target_network()

    print("Done.")
import numpy as np
import torch
import d3rlpy
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

class TabularPolicyIteration:
    """
    Implements the 'AI Clinician' approach (Komorowski et al., 2018).
    
    1. Discretizes continuous patient states into clusters (K-Means).
    2. Builds a tabular MDP (Transition Matrix T and Reward Matrix R).
    3. Solves for the Optimal Policy using Policy Iteration.
    """
    def __init__(self, n_states=750, n_actions=25, gamma=0.99, device='cpu'):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.device = device
        
        # The clustering model (The "State Encoder")
        self.kmeans = MiniBatchKMeans(n_clusters=n_states, random_state=42, batch_size=256)
        
        # MDP Matrices (Will be filled in .fit())
        self.T = None # Transition Probabilities: P(s' | s, a)
        self.R = None # Mean Rewards: R(s, a)
        
        # Value functions
        self.V = None # State Value
        self.Q = None # Action Value
        self.policy = None # Deterministic Policy

    def fit(self, dataset: d3rlpy.dataset.MDPDataset, n_iterations=100):
        """
        Fits the MDP to the patient trajectories and runs Policy Iteration.
        """
        observations = dataset.observations
        actions = dataset.actions.flatten().astype(int)
        rewards = dataset.rewards.flatten()
        terminals = dataset.terminals.flatten()

        print("1. Discretizing State Space (Clustering)...")
        # In the paper, they clustered physiological features to create discrete states
        self.kmeans.fit(observations)
        discrete_states = self.kmeans.predict(observations)

        # We need next_states for transitions. 
        # d3rlpy datasets are flat, so we shift indices.
        # We must handle terminals correctly (next state is invalid if terminal).
        print("2. Constructing MDP Matrices (T and R)...")
        
        # Initialize T (Count) and R (Sum)
        T_counts = torch.zeros((self.n_states, self.n_actions, self.n_states), device=self.device)
        R_sums = torch.zeros((self.n_states, self.n_actions), device=self.device)
        SA_counts = torch.zeros((self.n_states, self.n_actions), device=self.device) + 1e-5 # Avoid div/0

        # Convert to tensor for speed
        s_t = torch.tensor(discrete_states[:-1], device=self.device, dtype=torch.long)
        a_t = torch.tensor(actions[:-1], device=self.device, dtype=torch.long)
        r_t = torch.tensor(rewards[:-1], device=self.device, dtype=torch.float32)
        s_tp1 = torch.tensor(discrete_states[1:], device=self.device, dtype=torch.long)
        dones = torch.tensor(terminals[:-1], device=self.device, dtype=torch.bool)

        # Vectorized Accumulation is hard, doing standard loop with TQDM for clarity/safety
        # (This takes a few seconds for ~100k transitions)
        batch_size = len(s_t)
        for i in tqdm(range(batch_size)):
            if dones[i]: continue # No transition recorded for terminal steps in this simple model
            
            curr_s = s_t[i]
            curr_a = a_t[i]
            next_s = s_tp1[i]
            rew = r_t[i]

            T_counts[curr_s, curr_a, next_s] += 1
            R_sums[curr_s, curr_a] += rew
            SA_counts[curr_s, curr_a] += 1

        # Normalize to get Probabilities and Means
        self.T = T_counts / SA_counts.unsqueeze(2) # Broadcasting
        self.R = R_sums / SA_counts

        print("3. Running Policy Iteration...")
        self._policy_iteration(max_iter=n_iterations)
        print("Done.")

    def _policy_iteration(self, max_iter=100):
        # Initialize random policy and values
        self.V = torch.zeros(self.n_states, device=self.device)
        self.Q = torch.zeros((self.n_states, self.n_actions), device=self.device)
        
        # Start with a random policy (indices of actions)
        current_policy = torch.randint(0, self.n_actions, (self.n_states,), device=self.device)

        for i in range(max_iter):
            # --- Policy Evaluation (Solve V for current policy) ---
            # Ideally: V = (I - gamma * P_pi)^-1 * R_pi
            # We approximate via iterative updates for stability
            for _ in range(50): # Inner eval loop
                # Select T and R based on current policy
                # Gather is tricky in 3D, we'll use advanced indexing
                row_indices = torch.arange(self.n_states, device=self.device)
                
                # T_pi: [n_states, n_states]
                T_pi = self.T[row_indices, current_policy[row_indices], :]
                # R_pi: [n_states]
                R_pi = self.R[row_indices, current_policy[row_indices]]
                
                # Bellman Expectation Equation: V = R + gamma * T * V_next
                self.V = R_pi + self.gamma * torch.matmul(T_pi, self.V)

            # --- Policy Improvement (Greedy maximization) ---
            # Q(s,a) = R(s,a) + gamma * sum(T(s,a,s') * V(s'))
            # Calculate Q for ALL actions
            self.Q = self.R + self.gamma * torch.matmul(self.T, self.V)
            
            # Find new greedy policy
            new_policy = torch.argmax(self.Q, dim=1)
            
            # Check convergence
            n_changes = (new_policy != current_policy).sum().item()
            if n_changes == 0:
                print(f"Converged at iteration {i}")
                break
            current_policy = new_policy

        self.policy = current_policy

    def predict(self, x):
        """
        Returns the optimal action for a list/batch of continuous observations.
        """
        # 1. Discretize input
        discrete_s = self.kmeans.predict(x)
        discrete_s_tensor = torch.tensor(discrete_s, device=self.device, dtype=torch.long)
        
        # 2. Lookup Policy
        optimal_actions = self.policy[discrete_s_tensor]
        return optimal_actions.cpu().numpy()

    def predict_value(self, x, action):
        """
        Returns Q(s, a) for given states and actions.
        """
        discrete_s = self.kmeans.predict(x)
        s_tensor = torch.tensor(discrete_s, device=self.device, dtype=torch.long)
        a_tensor = torch.tensor(action, device=self.device, dtype=torch.long)
        
        # Lookup Q table
        q_vals = self.Q[s_tensor, a_tensor]
        return q_vals.cpu().numpy()

# ==========================================
# USAGE EXAMPLE
# ==========================================

# 1. Create Dummy Data (representing your patient data)
# 1000 patients, 50 timesteps, 10 physiological features
N_SAMPLES = 50000
OBS_DIM = 10
N_ACTIONS = 25 # (e.g., 5x5 grid of IV fluids and Vasopressors)

observations = np.random.random((N_SAMPLES, OBS_DIM)).astype(np.float32)
actions = np.random.randint(0, N_ACTIONS, N_SAMPLES).astype(np.float32) # d3rlpy expects floats
rewards = np.random.choice([-1.0, 0.0, 1.0], N_SAMPLES).astype(np.float32)
terminals = np.zeros(N_SAMPLES).astype(np.float32)
terminals[::50] = 1.0 # End episode every 50 steps

# Create Standard d3rlpy Dataset
dataset = d3rlpy.dataset.MDPDataset(observations, actions, rewards, terminals)

# 2. Initialize the AI Clinician Implementation
# The paper used 750 clusters for sepsis data
model = TabularPolicyIteration(
    n_states=750, 
    n_actions=N_ACTIONS, 
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# 3. Fit the MDP (Clustering -> Matrix Build -> Solve)
model.fit(dataset)

# 4. Predict Treatment Strategy
# New patient state
new_patient = np.random.random((1, OBS_DIM)).astype(np.float32)

# Get optimal action (ID 0-24)
optimal_action = model.predict(new_patient)
print(f"Optimal Action ID: {optimal_action}")

# Get Expected Outcome (Q-value)
predicted_outcome = model.predict_value(new_patient, optimal_action)
print(f"Predicted Outcome Value: {predicted_outcome}")
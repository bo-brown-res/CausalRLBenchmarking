import torch
import torch.nn as nn
import torch.optim as optim
import d3rlpy

from dataloading.make_datasets import select_dataset
from fitting.fit_causal_model import setup_and_run_cs
from methods.causal_based.DragonNet import DragonNet
from methods.causal_based.TARNet import TARNet

class FQENetwork(nn.Module):
    def __init__(self, num_covariates, num_treatments, hidden_units=64):
        super(FQENetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_covariates, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, num_treatments) # Output Q-value for each treatment
        )

    def forward(self, x):
        return self.net(x)
		
		
def get_tarnet_action(model, covariates):
    """
    Passes covariates through TARNet and returns the action (treatment) index 
    with the highest predicted value.
    """
    model.eval() # Ensure model is in eval mode
    with torch.no_grad():
        # model(covariates) returns a list of tensors
        outs = model(covariates) 
        
        # Stack list into tensor: [batch_size, num_treatments, num_outputs]
        # We squeeze the last dim assuming num_outputs=1 (scalar reward/outcome)
        outs_stack = torch.stack(outs, dim=1).squeeze(-1)
        
        # Get the index (action) with the highest value
        best_actions = torch.argmax(outs_stack, dim=1)
        return best_actions
		
		
def run_fitted_q_evaluation(tarnet_model, dataset, num_epochs=100, gamma=0.99):
    """
    Args:
        tarnet_model: Your trained TARNet instance.
        dataset: Your 'mydata' object (assumed to be a PyTorch DataLoader or similar).
        num_epochs: How many times to iterate over the dataset.
        gamma: Discount factor.
    """
    
    # 1. Initialize the FQE Network (The Evaluator)
    # Assuming we can get dimensions from the tarnet model
    num_covariates = tarnet_model.shared_net[0].in_features
    num_treatments = tarnet_model.num_treatments

    model_device = next(tarnet_model.parameters()).device
    
    fqe_net = FQENetwork(num_covariates, num_treatments).to(model_device)
    optimizer = optim.Adam(fqe_net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # 2. Training Loop
    print("Starting FQE Training...")
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        # We assume dataset yields batches of: (obs, action, reward, next_obs, done)
        for batch in dataset:
            # Unpack batch (ensure they are tensors)
            # obs: [batch, num_covariates]
            # action: [batch] (indices of treatments taken)
            # reward: [batch]
            # next_obs: [batch, num_covariates]
            # done: [batch] (0 if sequence continues, 1 if ended)
            obs, action, reward, done, next_obs, masks = batch
            
            next_obs = next_obs.to(model_device)
            reward = reward.to(model_device)
            done = done.to(model_device)
            obs = obs.to(model_device)
            action = action.to(model_device)
            # --- STEP A: Calculate Target (The "Truth") ---
            with torch.no_grad():
                # 1. Ask TARNet what action it WOULD take in the next state
                next_action_from_policy = get_tarnet_action(tarnet_model, next_obs)
                
                # 2. Ask FQE Net for the value of ALL actions in next state
                all_next_q_values = fqe_net(next_obs)
                
                # 3. Select the Q-value corresponding to TARNet's chosen action
                # gather requires indices to have same dim as src, so we unsqueeze
                target_q_value = all_next_q_values.gather(2, next_action_from_policy.unsqueeze(1)).squeeze(1)
                
                # 4. Compute Bellman Target
                # If done, target is just reward. Otherwise, reward + gamma * future
                if len(done.shape) == 2:
                    done = done.unsqueeze(-1)

                vitem = gamma * target_q_value
                vitem = torch.concat([torch.zeros(vitem.shape[0], 1).to(model_device), vitem], dim=-1).unsqueeze(-1)
                target = reward + (vitem * (1 - done))

            # --- STEP B: Train FQE Network ---
            # 1. Predict Q-values for current state
            current_q_values = fqe_net(obs)
            
            # 2. Select Q-value for the action that was ACTUALLY taken in the dataset
            # Note: We use the actual dataset action here, not the model's action
            predicted_q = current_q_values.gather(1, action.argmax(-1).unsqueeze(-1)).squeeze(1)
            
            # 3. Update weights
            loss = loss_fn(predicted_q, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Avg Loss = {total_loss / len(dataset):.4f}")

    print("FQE Training Complete.")
    return fqe_net
	
	
#4. How to Interpret the Result
#Once run_fitted_q_evaluation finishes, you have a trained fqe_net. To actually "score" your model, you estimate the expected value of the starting states in your dataset.

def evaluate_policy_value(fqe_net, tarnet_model, initial_states):
    """
    Returns the estimated expected return of the TARNet policy.
    initial_states: A batch of starting states from your dataset.
    """
    fqe_net.eval()
    with torch.no_grad():
        # 1. What action does TARNet take at the start?
        actions = get_tarnet_action(tarnet_model, initial_states)
        
        # 2. What is the Q-value of that action?
        q_values = fqe_net(initial_states)
        value_estimates = q_values.gather(1, actions.unsqueeze(1))
        
        # 3. Average over the batch
        return value_estimates.mean().item()

def run_FQE(modeltype):
    train_config = {
        'device': 'cuda:0',
        'batch_size': 512,
        'learning_rate': 1e-5,
        'discount_factor': 1.0, #0.99,
        'n_critics': 2,
        'alpha': 0.1,
        'mask_size': 10,
        'n_steps': 3,#20000, #200000, #3000,
        'n_steps_per_epoch': 1000, #1000, #1000,
        'initial_temperature': 0.1,
        'lambda_alpha': 1.0,
        #CRN params
        'rnn_hidden_units': 128, #256,
        'fc_hidden_units': 64, # 128
        #DragonNet / TARNet
        'hidden_units': 128,
        'dragon_alpha': 1.0,        
    }

    data_config ={
        'batch_size': 512,
        'max_seq_len': 24*7,
        # 'targets': args.targets, #'reward', # 'return', '1-step-return'
        'target_value': 'final_sum',
        'reward_scaler': 1.0,
        'state_masking_p': 1.0,
    }
    if modeltype == 'RL':
        data_config['targets'] = '8-step-return'
        train_config['targets'] = '8-step-return'
    if modeltype == 'CS':
        data_config['targets'] = 'reward'
        train_config['targets'] = 'reward'

    for dsname in ['mimic4_hourly']: #['mimic4_hourly']:
        if dsname == 'inspire_hourly':
            n_epochs = 75
            rl_model_path = "/mnt/d/research/rl_causal/finalproject/fqe_models/SoftActorCritic_8-step-return_final_sum_123_p=1.0/inspire_model/model_11000.d3"
        else:
            n_epochs = 95
            rl_model_path = "/mnt/d/research/rl_causal/finalproject/fqe_models/SoftActorCritic_8-step-return_final_sum_123_p=1.0/mimic_model/model_8000.d3"


        train_dataset, val_dataset, test_dataset, seperate_ites = select_dataset(
            ds_name=dsname, 
            val_size=0.1, 
            test_size=0.2,
            # subsample_frac=0.1, #TODO: remove when algos working
            fmt=modeltype,
            data_config=data_config,
        )

        if modeltype == 'CS':
            trained_model, train_results, test_results = setup_and_run_cs(
                method_name='TARNet',
                train_config=train_config,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                dataset_name=dsname,
                savetag="temp"
            )

            assert isinstance(trained_model, TARNet) or isinstance(trained_model, DragonNet), "FATAL ERROR"

            trained_fqe = run_fitted_q_evaluation(trained_model, train_dataset, num_epochs=n_epochs)

            initial_observations_batch = torch.concat([x[0][:, 0, :] for x in test_dataset]).to('cuda')
            
            score = evaluate_policy_value(trained_fqe, trained_model, initial_observations_batch)
            # n_time_points_to_amortize = test_dataset.dataset.data[3].shape[1]
            # print(f"Estimated Policy Value: {score * n_time_points_to_amortize}")
            print(f"Estimated Policy Value: {score}")

        elif modeltype == 'RL':
            fitted_algo = d3rlpy.load_learnable(rl_model_path)

            # setup FQE algorithm

            fqe = d3rlpy.ope.DiscreteFQE(algo=fitted_algo, config=d3rlpy.ope.FQEConfig())

            # start FQE training
            fqe.fit(
            train_dataset, #should be MDP dataset type
            n_steps=10000,
            n_steps_per_epoch=1000,
            evaluators={
                "test_init_value": d3rlpy.metrics.InitialStateValueEstimationEvaluator(episodes=test_dataset.episodes),
                "test_soft_opc": d3rlpy.metrics.SoftOPCEvaluator(180, episodes=test_dataset.episodes),  # set 180 for success return threshold here
                "test_average_value": d3rlpy.metrics.AverageValueEstimationEvaluator(episodes=test_dataset.episodes),
                "train_init_value": d3rlpy.metrics.InitialStateValueEstimationEvaluator(episodes=train_dataset.episodes),
                "train_soft_opc": d3rlpy.metrics.SoftOPCEvaluator(180, episodes=train_dataset.episodes),  # set 180 for success return threshold here
                "train_average_value": d3rlpy.metrics.AverageValueEstimationEvaluator(episodes=train_dataset.episodes),
            
            },
            )


            print("Finished!")
        else:
            raise NotImplementedError()
        

if __name__ == '__main__':
    run_FQE(modeltype='CS')
    # run_FQE(modeltype='RL')
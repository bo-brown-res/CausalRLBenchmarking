Set the current working directory to the finalproject/ folder

To train a causal model. run:

    python3 main.py

    Using the arguments:

    --method_name [CRN or DragonNet or TARNet]
    --dataset_name, [YOUR DATASET NAME, i.e. 'mimic4_-_hourly']
    --targets reward
    --target_value final_sum
    --savetag"  [YOUR LOGGING FOLDER]
    --reward_scaler  [1 OR 0.5 depending on experiment]
    --state_masking_p 1.0
    --random_seed 10000

To train an RL model, run: 

    python3 main.py

    Using the arguments:

    --method_name [CausalDQN or DQN or CQL or SoftActorCritic]
    --dataset_name, [YOUR DATASET NAME, i.e. 'mimic4_-_hourly']
    --targets ['k-step-return' where k is an int of your choosing]
    --target_value final_sum
    --savetag"  [YOUR LOGGING FOLDER]
    --reward_scaler  [1 OR 0.5 depending on experiment]
    --state_masking_p 1.0
    --random_seed 10000


To run the Reinforcement Learning algorithms and evaluate ITE on the synthetic datasets, run:

    python3 repeater.py

To run the Causal Machine Learning algorithms and evaluate ITE on the synthetic datasets, run:

    python3 repeater_causal.py

To run the FQE evaluations, move to the fqe.py file and chang the 'inspire_model_path' and 'mimic_model_path' variables to point to your trained models. Then run:

    python3 fqe.py
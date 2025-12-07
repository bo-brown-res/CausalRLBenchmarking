import d3rlpy
from d3rlpy.algos.causal_dqn import CausalDQNConfig
from d3rlpy.metrics.evaluators import TrueValueErrorEvaluator
import numpy as np

def run_causal_dqn_fit(
    train_dataset,
    observation_scaler,
    reward_scaler,
    action_scaler,
    optim_factory,
    shared_evalautor_dict,
):
    
    causal_dqn_algo = CausalDQNConfig(
        observation_scaler=observation_scaler,
        # action_scaler=action_scaler, #only for continuous actions
        reward_scaler=reward_scaler,
        optim_factory=optim_factory,
        batch_size=512,
        learning_rate=1e-3,
        gamma=0.99,
        n_critics=2,
        #causal dqn specific params
        alpha=0.1,
        mask_size=10,
    ).create(device="cuda:0")

    true_val_evalautor = TrueValueErrorEvaluator(
        get_true_state_val_fn=lambda obs: np.zeros(len(obs)),
        error_type='mae',
    )

    causal_dqn_algo.fit(
        dataset=train_dataset,
        n_steps=10000,
        n_steps_per_epoch=1000,
        # eval_episodes=val_dataset,
        evaluators=shared_evalautor_dict,
    )

    return causal_dqn_algo
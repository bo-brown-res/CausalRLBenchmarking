import d3rlpy
import numpy as np

from evaluate.custom_evaluators import TrueValueErrorEvaluator


def setup_and_run_rl(
    method_name,
    train_config,
    train_dataset,
    val_dataset,
    test_dataset,
):
    observation_scaler = d3rlpy.preprocessing.StandardObservationScaler()
    reward_scaler = d3rlpy.preprocessing.StandardRewardScaler()
    action_scaler = d3rlpy.preprocessing.MinMaxActionScaler()
    optim_factory = d3rlpy.optimizers.AdamFactory(
        lr_scheduler_factory=d3rlpy.optimizers.WarmupSchedulerFactory(
            warmup_steps=1000
        )
    )

    shared_evalautor_dict = {
            'VAL_td_error': d3rlpy.metrics.TDErrorEvaluator(episodes=val_dataset.episodes),
            'VAL_average_value': d3rlpy.metrics.AverageValueEstimationEvaluator(episodes=val_dataset.episodes),
            'VAL_true_value_error': TrueValueErrorEvaluator(
                get_true_state_val_fn=lambda obs: np.zeros(len(obs)),
                error_type='mae',
                episodes=val_dataset.episodes
            ),
            #want to keep track of test set performance during training as well, but not use for making decisions.
            'TEST_td_error': d3rlpy.metrics.TDErrorEvaluator(episodes=test_dataset.episodes),
            'TEST_average_value': d3rlpy.metrics.AverageValueEstimationEvaluator(episodes=test_dataset.episodes),
            'TEST_true_value_error': TrueValueErrorEvaluator(
                get_true_state_val_fn=lambda obs: np.zeros(len(obs)),
                error_type='mae',
                episodes=test_dataset.episodes
            ),
    }

    if method_name == 'causal_dqn':
        from fitting.rl_CausalDQN import run_causal_dqn_fit
        fitted_algo = run_causal_dqn_fit(
            train_config=train_config,
            train_dataset=train_dataset,
            observation_scaler=observation_scaler,
            reward_scaler=reward_scaler,
            action_scaler=action_scaler,
            optim_factory=optim_factory,
            shared_evalautor_dict=shared_evalautor_dict,
        )
    elif method_name == 'soft_actor_critic':
        from fitting.rl_CausalDQN import run_SAC_fit
        fitted_algo = run_SAC_fit(
            train_config=train_config,
            train_dataset=train_dataset,
            observation_scaler=observation_scaler,
            reward_scaler=reward_scaler,
            action_scaler=action_scaler,
            optim_factory=optim_factory,
            shared_evalautor_dict=shared_evalautor_dict,
        )
    else:
        raise ValueError(f"Unknown method name: {method_name}")
    
    fitted_algo.save('dqn_test_01.d3')
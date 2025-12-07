import d3rlpy
# import d4rl
# from d3rlpy.datasets import get_cartpole # CartPole-v0 dataset
# from d3rlpy.datasets import get_pendulum # Pendulum-v0 dataset
# from d3rlpy.datasets import get_atari    # Atari 2600 task datasets
# from d3rlpy.datasets import get_d4rl     # D4RL datasets
from d3rlpy.algos.qlearning.dqn import DQN, DQNConfig
from d3rlpy.metrics import TDErrorEvaluator
import pickle
import numpy as np
import pandas as pd
from torch.optim import Adam

from fitting.fit_rl_model import setup_and_run_rl
from methods.make_datasets import select_dataset

def main():
    argparser = d3rlpy.cli.create_argument_parser()
    args = argparser.parse_args()


    working_methods = {
        #rl-based methods
        'policy_iteration':  ['RL', xxx],
        'causal_dqn':        ['RL', xxx],
        'soft_actor_critic': ['RL', xxx],
        'proximal_rl': ['RL', xxx],

        #causal-based methods
        'causal_forest': ['CS', xxx],
        'tarnet': ['CS', xxx],
        'dragonnet': ['CS', xxx],
        'CRN': ['CS', xxx],
        'T4': ['CS', xxx],
        'gtransformer': ['CS', xxx],
    }

    m_name = args.method_name

    if m_name in working_methods:
        method_type, run_method_function = working_methods[m_name]
    else:
        raise ValueError(f"Unknown method name: {m_name}")
    
    #get the data
    train_dataset, val_dataset, test_dataset = select_dataset(ds_name='mimic4_hourly', val_size=0.1, test_size=0.2)

    if method_type == 'RL':
        setup_and_run_rl(
            method_type=m_name,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
        )
    elif method_type == 'CS':
        setup_and_run_cs(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
        )
    else:
        raise ValueError(f"Unknown method type: {method_type}")




if __name__ == "__main__":
    main()  
# sac = d3rlpy.algos.SACConfig(compile_graph=True).create(device="cuda:0")






##################################################################################
# observation_scaler = d3rlpy.preprocessing.StandardObservationScaler()
# reward_scaler = d3rlpy.preprocessing.StandardRewardScaler()
# action_scaler = d3rlpy.preprocessing.MinMaxActionScaler()


# optim_factory = d3rlpy.optimizers.AdamFactory(
#     lr_scheduler_factory=d3rlpy.optimizers.WarmupSchedulerFactory(
#         warmup_steps=1000
#     )
# )

# dqn = DQNConfig(
#     observation_scaler=observation_scaler,
#     # action_scaler=action_scaler, #only for continuous actions
#     reward_scaler=reward_scaler,
#     optim_factory=optim_factory,
#     batch_size=512,
#     learning_rate=1e-3,
#     gamma=0.99,
#     n_critics=2,
# ).create(device="cuda:0")


# # train offline
# dqn.fit(
#     dataset=train_dataset,
#     n_steps=10000,
#     n_steps_per_epoch = 1000,
#     # eval_episodes=test_dataset,
#     evaluators={
#         'td_error': d3rlpy.metrics.TDErrorEvaluator(episodes=test_dataset.episodes),
#         'value_scale': d3rlpy.metrics.AverageValueEstimationEvaluator(episodes=test_dataset.episodes),
#         # 'custom_evaluator': CustomEvaluator(),
#         'average_value': d3rlpy.metrics.AverageValueEstimationEvaluator(episodes=test_dataset.episodes),
#     },
# )
# dqn.save('dqn_test_01.d3')

# print(f"Here!")
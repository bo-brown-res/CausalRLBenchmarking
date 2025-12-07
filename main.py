import argparse
# import d3rlpy
# # import d4rl
# # from d3rlpy.datasets import get_cartpole # CartPole-v0 dataset
# # from d3rlpy.datasets import get_pendulum # Pendulum-v0 dataset
# # from d3rlpy.datasets import get_atari    # Atari 2600 task datasets
# # from d3rlpy.datasets import get_d4rl     # D4RL datasets
# from d3rlpy.algos.qlearning.dqn import DQN, DQNConfig
# from d3rlpy.metrics import TDErrorEvaluator
# import pickle
# import numpy as np
# import pandas as pd
# from torch.optim import Adam
import random
import numpy as np
import torch

from dataloading.make_datasets import select_dataset
from fitting.fit_causal_model import setup_and_run_cs
from fitting.setup_rl import setup_and_run_rl

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    argparser = argparse.ArgumentParser(description='Run RL or Causal method on dataset')
    argparser.add_argument(
        '--method_name', type=str, required=True,
        help='Name of the method to run. Options: policy_iteration, causal_dqn, soft_actor_critic, proximal_rl, causal_forest, tarnet, dragonnet, CRN, T4, gtransformer'
    )
    argparser.add_argument(
        '--dataset_name', type=str, default='mimic4_hourly',
        help='Name of the dataset to use. Options: mimic4_hourly'
    )
    args = argparser.parse_args()

    set_random_seed(123)

    working_methods = {
        #rl-based methods
        'policy_iteration':  ['RL', None],
        'causal_dqn':        ['RL', None],
        'soft_actor_critic': ['RL', None],
        # 'proximal_rl': ['RL', None],

        #causal-based methods
        'causal_forest': ['CS', None],
        'tarnet': ['CS', None],
        'dragonnet': ['CS', None],
        'CRN': ['CS', None],
        'T4': ['CS', None],
        'gtransformer': ['CS', None],
    }

    m_name = args.method_name

    if m_name in working_methods:
        method_type, run_method_function = working_methods[m_name]
    else:
        raise ValueError(f"Unknown method name: {m_name}")

    #define the training config
    train_config = {
        'device': 'cuda:0',
        'batch_size': 512,
        'learning_rate': 1e-3,
        'discount_factor': 0.99,
        'n_critics': 2,
        'alpha': 0.1,
        'mask_size': 10,
        'n_steps': 3000,
        'n_steps_per_epoch': 1000,
        'initial_temperature': 0.1,
        'lambda_alpha': 1.0,
    }

    #load dataset
    train_dataset, val_dataset, test_dataset = select_dataset(
        ds_name='mimic4_hourly', 
        val_size=0.1, 
        test_size=0.2,
        subsample_frac=0.1, #TODO: remove when algos working
        fmt=method_type,
        train_config=train_config
    )

    if method_type == 'RL':
        setup_and_run_rl(
            method_name=m_name,
            train_config=train_config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            fmt='CS'
        )
    elif method_type == 'CS':
        setup_and_run_cs(
            method_name=m_name,
            train_config=train_config,
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
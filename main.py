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
from models.factories import MyCustomEncoderFactory, MyCustomQFunctionFactory

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
    argparser.add_argument(
        '--targets', type=str, default='mimic4_hourly',
        help='TODO',
        # choices=['reward', 'return', '1-step-return']
    )    
    argparser.add_argument(
        '--target_value', type=str, default='mimic4_hourly',
        help='TODO',
        choices=['binary', 'plusminusone', 'cumulative', 'reals', 'finals', 'final_sum']
    )
    argparser.add_argument(
        '--savetag', type=str, default='',
        help='TODO',
    )
    argparser.add_argument(
        '--reward_scaler', type=float, default=1.0,
        help='TODO',
    )
    argparser.add_argument(
        '--random_seed', type=int, default=123,
        help='TODO',
    )
    argparser.add_argument(
        '--state_masking_p', type=float, default=1.0,
        help='TODO',
    )
    args = argparser.parse_args()
    set_random_seed(args.random_seed)

    valid_datasets = [
        "mimic4_hourly",
        "epicare_len12_acts4ep_10000",
    ]

    working_methods = {
    #rl-based methods
        'policy_iteration':  ['RL', None],
        'CausalDQN':        ['RL', None],
        'SoftActorCritic': ['RL', None],
        'CQL': ['RL', None],
        'IQL': ['RL', None],
        'DQN': ['RL', None],
        # 'proximal_rl': ['RL', None],
    #causal-based methods
        # 'causal_forest': ['CS', None],
        'TARNet': ['CS', None],
        'SequentialTARNet': ['CS', None],
        'DragonNet': ['CS', None],
        'CRN': ['CS', None],
        # 'T4': ['CS', None],
        # 'gtransformer': ['CS', None],
    }

    m_name = args.method_name
    if m_name in working_methods:
        method_type, run_method_function = working_methods[m_name]
    else:
        raise ValueError(f"Unknown method name: {m_name}")

    #define the training config
    batch_size = 512

    train_config = {
        'device': 'cuda:0',
        'batch_size': batch_size,
        'learning_rate': 1e-5,
        'discount_factor': 1.0, #0.99,
        'n_critics': 2,
        'alpha': 0.1,
        'mask_size': 10,
        'n_steps': 20000, #200000, #3000,
        'n_steps_per_epoch': 1000, #1000, #1000,
        'initial_temperature': 0.1,
        'lambda_alpha': 1.0,
        #CRN params
        'rnn_hidden_units': 128, #256,
        'fc_hidden_units': 64, # 128
        #DragonNet / TARNet
        'hidden_units': 128,
        'dragon_alpha': 1.0,
        'targets': args.targets,
        
    }
    # qfn_fact = MyCustomQFunctionFactory(hdim=128)
    enc_fact = MyCustomEncoderFactory(feature_size=64, hdim=128)
    train_config['encoder_factory'] = enc_fact#enc_fact
    # train_config['q_func_factory'] = None#qfn_fact
    curdir = "/mnt/d/research/rl_causal/finalproject/fqe_models/"
    train_config['logdir'] = f"{curdir}{args.method_name}_{args.targets}_{args.target_value}_{args.random_seed}_p={args.state_masking_p}"

    data_config ={
        'batch_size': batch_size,
        'max_seq_len': 24*7,
        'targets': args.targets, #'reward', # 'return', '1-step-return'
        'target_value': args.target_value, #'binary' # 'binary', 'plusminusone', 'cumulative', 'reals' 'final'
        'reward_scaler': args.reward_scaler,
        'state_masking_p': args.state_masking_p,
    }
    if method_type == 'rl':
        print(f"INFO - We are using RL, so must provide a full trajectory of reward values")
        assert 'reward' in data_config['targets'] or '-step-return' in data_config['targets'], f"ERROR FOR RL: data_config['targets'] = {data_config['targets']} not valid"

    #load dataset
    train_dataset, val_dataset, test_dataset,seperate_ites = select_dataset(
        ds_name=args.dataset_name, 
        val_size=0.1, 
        test_size=0.2,
        # subsample_frac=0.1, #TODO: remove when algos working
        fmt=method_type,
        data_config=data_config,
    )

    if method_type == 'RL':
        setup_and_run_rl(
            method_name=m_name,
            train_config=train_config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            seperate_ites=seperate_ites,
        )
    elif method_type == 'CS':
        trained_model, train_results, test_results = setup_and_run_cs(
            method_name=m_name,
            train_config=train_config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            dataset_name=args.dataset_name,
            savetag=args.savetag
        )
    else:
        raise ValueError(f"Unknown method type: {method_type}")


    print(f"Finished!")
    # TODO: DO something with trained_model, train_results, test_results


if __name__ == "__main__":
    main()  






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
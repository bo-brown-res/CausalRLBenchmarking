import d3rlpy
from methods.rl_based.CausalDQ import CausalDQNConfig



import torch
import numpy as np
import d3rlpy



################################################################################

def run_CausalDQN(
    train_config,
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
        # reward_scaler=reward_scaler,
        optim_factory=optim_factory,
        batch_size=train_config['batch_size'], #512
        learning_rate=train_config['learning_rate'], #1e-3,
        gamma=train_config['discount_factor'], #0.99,
        n_critics=train_config['n_critics'], #2,
        #causal dqn specific params
        alpha=train_config['alpha'], #0.1,
        mask_size=train_config['mask_size'], #10,
    ).create(device=train_config['device'])

    causal_dqn_algo.fit(
        dataset=train_dataset,
        n_steps=train_config['n_steps'], #10000,
        n_steps_per_epoch=train_config['n_steps_per_epoch'], #1000,
        # eval_episodes=val_dataset,
        evaluators=shared_evalautor_dict,
    )

    return causal_dqn_algo

################################################################################

def run_SAC(
    train_config,
    train_dataset,
    observation_scaler,
    reward_scaler,
    action_scaler,
    optim_factory,
    shared_evalautor_dict,
):
    
    sac_algo = d3rlpy.algos.DiscreteSACConfig(
        observation_scaler=observation_scaler,
        # reward_scaler=reward_scaler,
        actor_learning_rate=train_config['learning_rate'], #1e-3,
        critic_learning_rate=train_config['learning_rate'], #1e-3,
        actor_optim_factory=optim_factory,
        critic_optim_factory=optim_factory,
        batch_size=train_config['batch_size'], #512
        # learning_rate=train_config['learning_rate'], #1e-3,
        gamma=train_config['discount_factor'], #0.99,
        n_critics=train_config['n_critics'], #2,
        initial_temperature=train_config['initial_temperature'], #0.1,
    ).create(device=train_config['device'])

    sac_algo.fit(
        dataset=train_dataset,
        n_steps=train_config['n_steps'], #10000,
        n_steps_per_epoch=train_config['n_steps_per_epoch'], #1000,
        # eval_episodes=val_dataset,
        evaluators=shared_evalautor_dict,
    )

    return sac_algo

################################################################################

def run_DQN(
    train_config,
    train_dataset,
    observation_scaler,
    reward_scaler,
    action_scaler,
    optim_factory,
    shared_evalautor_dict,
):
    
    dqn_algo = d3rlpy.algos.DQNConfig(
        observation_scaler=observation_scaler,
        # action_scaler=action_scaler, #only for continuous actions
        # reward_scaler=reward_scaler,
        optim_factory=optim_factory,
        batch_size=train_config['batch_size'], #512
        learning_rate=train_config['learning_rate'], #1e-3,
        gamma=train_config['discount_factor'], #0.99,
        n_critics=train_config['n_critics'], #2,
        # causal dqn specific params
        # alpha=train_config['alpha'], #0.1,
        # mask_size=train_config['mask_size'], #10,
    ).create(device=train_config['device'])

    dqn_algo.fit(
        dataset=train_dataset,
        n_steps=train_config['n_steps'], #10000,
        n_steps_per_epoch=train_config['n_steps_per_epoch'], #1000,
        # eval_episodes=val_dataset,
        evaluators=shared_evalautor_dict,
    )

    return dqn_algo


import pickle

import d3rlpy
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from dataloading.variablelength_data import VariableLengthDataset, pack_collate


def get_data_items(data_object, subsample_frac=None):
    observations = data_object['states']
    actions = data_object['actions']
    rewards = data_object['rewards']
    terminals = data_object['terminals']
    actionmap = data_object.get('actionmap', None)
    colnames = data_object.get('colnames', None)
    threshold_hours = data_object.get('threshold_hours', None)
    true_ites = data_object.get('true_ites', None)

    if subsample_frac is not None:
        n_episodes = len(observations)
        subsample_size = int(n_episodes * subsample_frac)
        selected_indices = np.random.choice(n_episodes, subsample_size, replace=False)

        observations = [observations[i] for i in selected_indices]
        actions = [actions[i] for i in selected_indices]
        rewards = [rewards[i] for i in selected_indices]
        terminals = [terminals[i] for i in selected_indices]
        if true_ites:
            true_ites = [true_ites[i] for i in selected_indices]

    return observations, actions, rewards, terminals, true_ites, actionmap, colnames, threshold_hours


def select_dataset(ds_name, val_size=0.1, test_size=0.2, subsample_frac=None, fmt='RL', train_config=None):
    if ds_name == "mimic4_hourly":
        mimic4_hourly_data = pickle.load(open("/mnt/d/research/rl_causal/notebooks/mimic4_hourly_datapackage.pkl", "rb"))
        obs, acts, rwds, terms, true_ites, amap, cnames, thhrs = get_data_items(mimic4_hourly_data, subsample_frac=subsample_frac)
    elif ds_name == "epicare_len12_acts4ep_10000":
        epicare_len12_acts4ep_10000 = pickle.load(open("/mnt/d/research/rl_causal/finalproject/data/epicare_len12_acts4ep_10000.pkl", "rb"))
        obs, acts, rwds, terms, true_ites, amap, cnames, thhrs = get_data_items(epicare_len12_acts4ep_10000, subsample_frac=subsample_frac)
    else:
        raise ValueError(f"Dataset {ds_name} not recognized.")
    

    train_dataset, val_dataset, test_dataset, seperate_ites = build_train_test_datasets(
        dformat=fmt,
        observations=obs, 
        actions=acts, 
        rewards=rwds, 
        terminals=terms, 
        true_ites=true_ites,
        test_size=test_size, 
        validation_size=val_size,
        train_config=train_config,
    )
    return train_dataset, val_dataset, test_dataset, seperate_ites


def build_train_test_datasets(dformat, observations, actions, rewards, terminals, true_ites, test_size=0.2, validation_size=0.0, train_config=None):
    n_episodes = len(observations)
    random_indices = np.random.permutation(n_episodes)
    n_train_episodes = int(n_episodes * (1 - test_size))

    train_indices = random_indices[:n_train_episodes]
    val_indices = []
    n_val_episodes = 0

    seperate_ites = {}

    if validation_size > 0.0:
        n_val_episodes = int(n_episodes * validation_size)
        val_indices = random_indices[n_train_episodes:n_train_episodes + n_val_episodes]
    test_indices = random_indices[n_train_episodes + n_val_episodes:]

    if dformat == 'RL':
        train_dataset = d3rlpy.dataset.MDPDataset(
            observations=np.concat([observations[i] for i in train_indices],axis=0),
            actions=np.concat([actions[i] for i in train_indices],axis=0),
            rewards=np.concat([rewards[i] for i in train_indices],axis=0),
            terminals=np.concat([terminals[i] for i in train_indices],axis=0),
        )
        seperate_ites['train'] = np.concat([true_ites[i] for i in train_indices],axis=0)

        test_dataset = d3rlpy.dataset.MDPDataset(
            observations=np.concat([observations[i] for i in test_indices],axis=0),
            actions=np.concat([actions[i] for i in test_indices],axis=0),
            rewards=np.concat([rewards[i] for i in test_indices],axis=0),
            terminals=np.concat([terminals[i] for i in test_indices],axis=0),
        )
        seperate_ites['test'] = np.concat([true_ites[i] for i in test_indices],axis=0)

        val_dataset = None
        if validation_size > 0.0:
            val_dataset = d3rlpy.dataset.MDPDataset(
                observations=np.concat([observations[i] for i in val_indices],axis=0),
                actions=np.concat([actions[i] for i in val_indices],axis=0),
                rewards=np.concat([rewards[i] for i in val_indices],axis=0),
                terminals=np.concat([terminals[i] for i in val_indices],axis=0),
            )
            seperate_ites['val'] = np.concat([true_ites[i] for i in val_indices],axis=0)

    elif dformat == 'CS':
        max_seq_len = train_config.get('max_seq_len', None)
        if max_seq_len is not None:
            observations = [obs[-max_seq_len:] for obs in observations]
            actions = [act[-max_seq_len:] for act in actions]
            rewards = [rwd[-max_seq_len:] for rwd in rewards]
            terminals = [trm[-max_seq_len:] for trm in terminals]

        #TODO: should rewards be flat? I.e. one per sequence instead of per time step?
        train_vardat = VariableLengthDataset(observations=[torch.from_numpy(observations[i]).to(torch.float32) for i in train_indices],
                                              actions=[torch.from_numpy(actions[i]).to(torch.float32) for i in train_indices],
                                              rewards=[torch.from_numpy(rewards[i]).to(torch.float32) for i in train_indices],
                                              terminals=[torch.from_numpy(terminals[i]).to(torch.float32) for i in train_indices],
                                              true_ites=[torch.from_numpy(true_ites[i]).to(torch.float32) for i in train_indices],
                                              )
        train_dataset = DataLoader(train_vardat, batch_size=train_config.get('batch_size'), collate_fn=pack_collate, shuffle=True)
        
        test_vardat = VariableLengthDataset(observations=[torch.from_numpy(observations[i]).to(torch.float32) for i in test_indices],
                                              actions=[torch.from_numpy(actions[i]).to(torch.float32) for i in test_indices],
                                              rewards=[torch.from_numpy(rewards[i]).to(torch.float32) for i in test_indices],
                                              terminals=[torch.from_numpy(terminals[i]).to(torch.float32) for i in test_indices],
                                              true_ites=[torch.from_numpy(true_ites[i]).to(torch.float32) for i in test_indices],
                                              )
        test_dataset = DataLoader(test_vardat, batch_size=train_config.get('batch_size'), collate_fn=pack_collate, shuffle=True)

        val_dataset = None
        if validation_size > 0.0:
            val_vardat = VariableLengthDataset(observations=[torch.from_numpy(observations[i]).to(torch.float32) for i in val_indices],
                                                        actions=[torch.from_numpy(actions[i]).to(torch.float32) for i in val_indices],
                                                        rewards=[torch.from_numpy(rewards[i]).to(torch.float32) for i in val_indices],
                                                        terminals=[torch.from_numpy(terminals[i]).to(torch.float32) for i in val_indices],
                                                        true_ites=[torch.from_numpy(true_ites[i]).to(torch.float32) for i in val_indices],
                                                        )
            val_dataset = DataLoader(val_vardat, batch_size=train_config.get('batch_size'), collate_fn=pack_collate, shuffle=True)
        
    else:
        raise ValueError(f"Unknown dataset format: {dformat}")


    return train_dataset, val_dataset, test_dataset, seperate_ites
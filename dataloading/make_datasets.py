

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
    actionmap = data_object['actionmap']
    colnames = data_object['colnames']
    threshold_hours = data_object['threshold_hours']

    if subsample_frac is not None:
        n_episodes = len(observations)
        subsample_size = int(n_episodes * subsample_frac)
        selected_indices = np.random.choice(n_episodes, subsample_size, replace=False)

        observations = [observations[i] for i in selected_indices]
        actions = [actions[i] for i in selected_indices]
        rewards = [rewards[i] for i in selected_indices]
        terminals = [terminals[i] for i in selected_indices]

    return observations, actions, rewards, terminals, actionmap, colnames, threshold_hours


def select_dataset(ds_name, val_size=0.1, test_size=0.2, subsample_frac=None, fmt='RL', train_config=None):
    if ds_name == "mimic4_hourly":
        mimic4_hourly_data = pickle.load(open("/mnt/d/research/rl_causal/notebooks/mimic4_hourly_datapackage.pkl", "rb"))
        obs, acts, rwds, terms, amap, cnames, thhrs = get_data_items(mimic4_hourly_data, subsample_frac=subsample_frac)
    else:
        raise ValueError(f"Dataset {ds_name} not recognized.")
    

    train_dataset, val_dataset, test_dataset = build_train_test_datasets(
        dformat=fmt,
        observations=obs, 
        actions=acts, 
        rewards=rwds, 
        terminals=terms, 
        test_size=test_size, 
        validation_size=val_size,
        batch_size=train_config.get('batch_size', 128)
    )
    return train_dataset, val_dataset, test_dataset


def build_train_test_datasets(dformat, observations, actions, rewards, terminals, test_size=0.2, validation_size=0.0, batch_size=None):
    n_episodes = len(observations)
    random_indices = np.random.permutation(n_episodes)
    n_train_episodes = int(n_episodes * (1 - test_size))

    train_indices = random_indices[:n_train_episodes]
    val_indices = []
    n_val_episodes = 0

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
        test_dataset = d3rlpy.dataset.MDPDataset(
            observations=np.concat([observations[i] for i in test_indices],axis=0),
            actions=np.concat([actions[i] for i in test_indices],axis=0),
            rewards=np.concat([rewards[i] for i in test_indices],axis=0),
            terminals=np.concat([terminals[i] for i in test_indices],axis=0),
        )
        val_dataset = None
        if validation_size > 0.0:
            val_dataset = d3rlpy.dataset.MDPDataset(
                observations=np.concat([observations[i] for i in val_indices],axis=0),
                actions=np.concat([actions[i] for i in val_indices],axis=0),
                rewards=np.concat([rewards[i] for i in val_indices],axis=0),
                terminals=np.concat([terminals[i] for i in val_indices],axis=0),
            )
    elif dformat == 'CS':
        #TODO: should rewards be flat? I.e. one per sequence instead of per time step?
        train_vardat = VariableLengthDataset(observations=[torch.from_numpy(observations[i]) for i in train_indices],
                                              actions=[torch.from_numpy(actions[i]) for i in train_indices],
                                              rewards=[torch.from_numpy(rewards[i]) for i in train_indices],
                                              terminals=[torch.from_numpy(terminals[i]) for i in train_indices])
        train_dataset = DataLoader(train_vardat, batch_size=batch_size, collate_fn=pack_collate, shuffle=True)
        
        test_vardat = VariableLengthDataset(observations=[torch.from_numpy(observations[i]) for i in test_indices],
                                              actions=[torch.from_numpy(actions[i]) for i in test_indices],
                                              rewards=[torch.from_numpy(rewards[i]) for i in test_indices],
                                              terminals=[torch.from_numpy(terminals[i]) for i in test_indices])
        test_dataset = DataLoader(test_vardat, batch_size=batch_size, collate_fn=pack_collate, shuffle=True)

        val_dataset = None
        if validation_size > 0.0:
            val_vardat = VariableLengthDataset(observations=[torch.from_numpy(observations[i]) for i in val_indices],
                                                        actions=[torch.from_numpy(actions[i]) for i in val_indices],
                                                        rewards=[torch.from_numpy(rewards[i]) for i in val_indices],
                                                        terminals=[torch.from_numpy(terminals[i]) for i in val_indices])
            val_dataset = DataLoader(val_vardat, batch_size=batch_size, collate_fn=pack_collate, shuffle=True)
        
    else:
        raise ValueError(f"Unknown dataset format: {dformat}")


    return train_dataset, val_dataset, test_dataset
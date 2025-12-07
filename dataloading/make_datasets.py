

import pickle

import d3rlpy
import numpy as np


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


def select_dataset(ds_name, val_size=0.1, test_size=0.2, subsample_frac=None):
    if ds_name == "mimic4_hourly":
        mimic4_hourly_data = pickle.load(open("/mnt/d/research/rl_causal/notebooks/mimic4_hourly_datapackage.pkl", "rb"))
        obs, acts, rwds, terms, amap, cnames, thhrs = get_data_items(mimic4_hourly_data, subsample_frac=subsample_frac)
    else:
        raise ValueError(f"Dataset {ds_name} not recognized.")
    
    train_dataset, val_dataset, test_dataset = build_train_test_datasets(
        observations=obs, 
        actions=acts, 
        rewards=rwds, 
        terminals=terms, 
        test_size=test_size, 
        validation_size=val_size
    )
    return train_dataset, val_dataset, test_dataset


def build_train_test_datasets(observations, actions, rewards, terminals, test_size=0.2, validation_size=0.0):
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


    return train_dataset, val_dataset, test_dataset

# train_dataset, _, test_dataset = build_train_test_datasets(
#     observations, actions, rewards, terminals, test_size=0.2, validation_size=0.0
# )
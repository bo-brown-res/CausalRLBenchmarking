import d3rlpy

import numpy as np
from d3rlpy.metrics import EvaluatorProtocol

class TrueValueErrorEvaluator(EvaluatorProtocol):
    def __init__(self, get_true_state_val_fn, error_type='mae', episodes=None):
        self.get_true_state_val_fn = get_true_state_val_fn
        self.error_type = error_type
        self.episodes = episodes

    def __call__(self, algo, test_dataset):
        episodes = self.episodes if self.episodes else test_dataset.episodes
        errors = []

        # Iterate over all episodes in the provided test_data
        for episode in episodes:
            observations = episode.observations
            actions = episode.actions
            realized_obs_actionval = algo.predict_value(observations, actions)

            # all_qvals = get_all_qvals_for_obs(algo=algo, obs=observations)
            true_values = self.get_true_state_val_fn(observations)
            
            if self.error_type == 'mae':
                batch_errors = np.abs(realized_obs_actionval - true_values)
            elif self.error_type == 'mse':
                batch_errors = (realized_obs_actionval - true_values) ** 2
            
            errors.extend(batch_errors)

        return np.mean(errors)
    



import numpy as np
from d3rlpy.metrics import EvaluatorProtocol
from evaluate.action_val_eval import compute_true_ite_error
from evaluate.wrappers import ITECalculationWrapper

class TrueValueErrorEvaluator(EvaluatorProtocol):
    def __init__(self, get_true_state_val_fn, true_ites, error_type='mae', episodes=None):
        self.get_true_state_val_fn = get_true_state_val_fn
        self.error_type = error_type
        self.dataset = episodes
        self.true_ites = true_ites

    def __call__(self, algo, test_dataset):
        # episodes = self.episodes if self.episodes else test_dataset.episodes
        # errors = []

        # # Iterate over all episodes in the provided test_data
        # for episode in episodes:
        #     observations = episode.observations
        #     actions = episode.actions
        #     realized_obs_actionval = algo.predict_value(observations, actions)

        #     # all_qvals = get_all_qvals_for_obs(algo=algo, obs=observations)
        #     true_values = self.get_true_state_val_fn(observations)
            
        #     if self.error_type == 'mae':
        #         batch_errors = np.abs(realized_obs_actionval - true_values)
        #     elif self.error_type == 'mse':
        #         batch_errors = (realized_obs_actionval - true_values) ** 2
            
        #     errors.extend(batch_errors)

        # return np.mean(errors)
    

        ite_wrapper = ITECalculationWrapper(algo)
        res = compute_true_ite_error(model=ite_wrapper, 
                            treatments=None, 
                            covariates=None, 
                            true_effects=self.true_ites, 
                            rldataset=self.dataset,
                            fmt='rl_test')
    
        return res
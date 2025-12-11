   

import numpy as np
import torch


class ITECalculationWrapper():
    def __init__(self, fitted_algo):
        self.fitted_ago = fitted_algo

    def predict_treatment_effect(self, treatment, covariates, rldataset, approach='E[G]_temporal_diff', **kwargs):
        observations = [x.observations for x in rldataset.episodes]
        actions = [x.actions for x in rldataset.episodes]

        # 2. Compute Q-values for ALL actions at each state
        # algo.predict_value returns (N, Num_Actions)
        episode_taken_a_vals = []
        with torch.no_grad():
            for o, a in zip(observations, actions):
                pair_q_values = self.fitted_ago.predict_value(o, a)

                #note: this gives values which are discounted sum of expected RETURN, not effect of current action
                # G(t:t+1) = (r_t) + (gamma * r_t+1) + (gamma^2 * E[G(t+2:T)])
                # G(t:t+2) = (r_t) + (gamma * r_t+1) + (gamma^2 * r_t+1) + (gamma^3 * E[G(t+2:T)])
                if approach == 'E[G]_temporal_diff':
                    pair_q_values[:-1] = pair_q_values[:-1] - pair_q_values[1:]
                
                episode_taken_a_vals.append(pair_q_values)
            
            # # Ensure it's a tensor for gathering
            # if isinstance(all_q_values, np.ndarray):
            #     all_q_values = torch.from_numpy(all_q_values).to(self.fitted_ago.device)

        # 3. Gather the Q-value for the specific action taken
        # We gather along dim=1 using the action indices
        # shape: (N, 1) -> squeeze to (N,)
        # taken_action_q_values = all_q_values.gather(1, actions.view(-1, 1)).squeeze(1)

        return episode_taken_a_vals
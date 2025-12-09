import numpy as np
import torch


# def get_all_qvals_for_obs(self, algo, obs:torch.Tensor) -> torch.Tensor:
#     assert obs.dim() == 2  # (batch_size, obs_dim)
#     if obs.dim() == 1:
#         obs = obs.unsqueeze(0)  # (1, obs_dim)

#     #get action val for all actions
#     all_actions = torch.arange(algo.action_size)
#     # Repeat the state for each action so we can query Q(s, a_0), Q(s, a_1)...
#     repeated_states = torch.tile(obs, (algo.action_size, 1))
#     # predict_value takes a batch of states and a batch of actions
#     q_values = algo.predict_value(repeated_states, all_actions)

#     return q_values


def compute_true_ite_error(model, treatments, covariates, true_effects, fmt='cs', **kwargs):
    # actions_taken = treatments
    val_of_acts_taken = model.predict_treatment_effect(treatments, covariates, **kwargs)

    if fmt == 'cs':
        val_of_acts_taken = val_of_acts_taken.squeeze()
        taken_acts_idxs = treatments.argmax(dim=-1)
        true_vals_of_acts_taken = torch.gather(true_effects, 2, taken_acts_idxs.unsqueeze(-1)).squeeze(-1)

    if 'rl_' in fmt:
        val_of_acts_taken = np.concatenate(val_of_acts_taken)
        if fmt == 'rl_test':
            true_effects = torch.from_numpy(true_effects)
            taken_acts_idxs = torch.from_numpy(np.concatenate([x.actions for x in kwargs['rldataset'].episodes]))
            true_vals_of_acts_taken = torch.gather(true_effects, 1, taken_acts_idxs).squeeze(-1).numpy()
    else:
        raise NotImplementedError()
    

    diff = (true_vals_of_acts_taken - val_of_acts_taken)**2
    mse = diff.mean()
    return mse
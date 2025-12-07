import torch


def get_all_qvals_for_obs(self, algo, obs:torch.Tensor) -> torch.Tensor:
    assert obs.dim() == 2  # (batch_size, obs_dim)
    if obs.dim() == 1:
        obs = obs.unsqueeze(0)  # (1, obs_dim)

    #get action val for all actions
    all_actions = torch.arange(algo.action_size)
    # Repeat the state for each action so we can query Q(s, a_0), Q(s, a_1)...
    repeated_states = torch.tile(obs, (algo.action_size, 1))
    # predict_value takes a batch of states and a batch of actions
    q_values = algo.predict_value(repeated_states, all_actions)

    return q_values


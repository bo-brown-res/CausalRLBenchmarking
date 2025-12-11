import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

class VariableLengthDataset(Dataset):
    def __init__(self, observations, actions, rewards, terminals, true_ites, fmt='3d_r_a'):

        torch_obs = torch.nn.utils.rnn.pad_sequence(observations, batch_first=True)
        torch_act = torch.nn.utils.rnn.pad_sequence(actions, batch_first=True)
        torch_rwd = torch.nn.utils.rnn.pad_sequence(rewards, batch_first=True)
        torch_trm = torch.nn.utils.rnn.pad_sequence(terminals, batch_first=True)
        torch_ite = torch.nn.utils.rnn.pad_sequence(true_ites, batch_first=True)

        lengths = [len(seq) for seq in observations]
        max_len = max(lengths)

        masks = torch.stack(
            [torch.concat([torch.ones(l), torch.zeros(max_len-l)]) if l < max_len else torch.ones(l) for l in lengths]
        )

        if fmt == '3d_r_a':
            if torch_act.ndim == 2 or torch_act.shape[-1] == 1:
                torch_act = torch.nn.functional.one_hot(torch_act.to(torch.long), num_classes=int(torch_act.max())+1) 
            if torch_rwd.ndim == 2:
                torch_rwd = torch_rwd.unsqueeze(-1)

        self.data = [
            torch_obs,
            torch_act,
            torch_rwd,
            torch_trm,
            torch_ite,
            masks,
            # observations,
            # actions,
            # rewards,
            # terminals,
            torch.tensor([len(seq) for seq in observations])
        ]

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        o = self.data[0][idx] #observation/state
        a = self.data[1][idx] #action
        r = self.data[2][idx] #reward
        t = self.data[3][idx] #terminal
        e = self.data[4][idx] #ites
        m = self.data[5][idx] #masks
        return o, a, r, t, e, m
    

def pack_collate(batch):
    # padded_obs = torch.stack([x[0] for x in batch])
    # padded_acts = torch.stack([x[1] for x in batch])
    # padded_rwds = torch.stack([x[2] for x in batch])
    # padded_trms = torch.stack([x[3] for x in batch])
    # dlengths = [x[4] for x in batch]

    # obs = torch.nn.utils.rnn.pack_padded_sequence(padded_obs, dlengths, batch_first=True, enforce_sorted=False)
    # acts = torch.nn.utils.rnn.pack_padded_sequence(padded_acts, dlengths, batch_first=True, enforce_sorted=False)
    # rwds = torch.nn.utils.rnn.pack_padded_sequence(padded_rwds, dlengths, batch_first=True, enforce_sorted=False)
    # term = torch.nn.utils.rnn.pack_padded_sequence(padded_trms, dlengths, batch_first=True, enforce_sorted=False)
           
    # obs = torch.nested.nested_tensor([x[0] for x in batch], layout=torch.jagged)
    # acts = torch.nested.nested_tensor([x[1] for x in batch], layout=torch.jagged)
    # rwds = torch.nested.nested_tensor([x[2] for x in batch], layout=torch.jagged)
    # term = torch.nested.nested_tensor([x[3] for x in batch], layout=torch.jagged)

    obs = torch.stack([x[0] for x in batch])
    acts = torch.stack([x[1] for x in batch])
    rwds = torch.stack([x[2] for x in batch])
    term = torch.stack([x[3] for x in batch])
    ites = torch.stack([x[4] for x in batch])
    masks = torch.stack([x[5] for x in batch])

    return obs, acts, rwds, term, ites, masks
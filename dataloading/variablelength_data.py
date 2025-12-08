import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

class VariableLengthDataset(Dataset):
    def __init__(self, observations, actions, rewards, terminals):
        self.data = [
            torch.nn.utils.rnn.pad_sequence(observations, batch_first=True),
            torch.nn.utils.rnn.pad_sequence(actions, batch_first=True),
            torch.nn.utils.rnn.pad_sequence(rewards, batch_first=True),
            torch.nn.utils.rnn.pad_sequence(terminals, batch_first=True),
            # observations,
            # actions,
            # rewards,
            # terminals,
            torch.tensor([len(seq) for seq in observations])
        ]

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        o = self.data[0][idx]
        a = self.data[1][idx]
        r = self.data[2][idx]
        t = self.data[3][idx]
        l = self.data[4][idx]
        return o, a, r, t, l
    

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

    return obs, acts, rwds, term
import torch
import torch.nn as nn
import torch.nn.functional as F

class TARNet(nn.Module):
    def __init__(self, num_covariates, num_treatments, num_outputs, hidden_units=64):
        super(TARNet, self).__init__()
        self.num_treatments = num_treatments
        
        self.shared_net = nn.Sequential(
            nn.Linear(num_covariates, hidden_units),
            nn.ELU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ELU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ELU()
        )

        for t in range(num_treatments):
            setattr(self, f'head{t}', nn.Sequential(
                nn.Linear(hidden_units, hidden_units),
                nn.ELU(),
                nn.Linear(hidden_units, hidden_units),
                nn.ELU(),
                nn.Linear(hidden_units, num_outputs)
            ))


    def forward(self, covariates, **kwargs):
        phi = self.shared_net(covariates)
        
        outs = []
        for t in range(self.num_treatments):
            outs.append(getattr(self, f'head{t}')(phi))

        return outs
    

    def predict_treatment_effect(self, treatments, covariates, **kwargs):
        ite_values = self(covariates, **kwargs)

        pred_stack = torch.stack(ite_values, dim=-1).squeeze()
        factual_predictions = torch.gather(pred_stack, 2, treatments.argmax(dim=-1).unsqueeze(-1)).squeeze()

        return factual_predictions
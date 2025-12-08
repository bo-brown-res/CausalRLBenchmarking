# https://arxiv.org/abs/1906.02120

import torch
import torch.nn as nn

class DragonNet(nn.Module):
    def __init__(self, num_covariates, num_treatments, num_outputs, hidden_units=64):
        super(DragonNet, self).__init__()
        self.num_treatments = num_treatments
        
        self.shared_base= nn.Sequential(
            nn.Linear(num_covariates, hidden_units),
            nn.ELU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ELU()
        )
        
        for t in range(num_treatments):
            setattr(self, f'head{t}', nn.Sequential(
                nn.Linear(hidden_units, hidden_units),
                nn.ELU(),
                nn.Linear(hidden_units, num_outputs)
            ))

        self.propensity_head = nn.Sequential(
            nn.Linear(hidden_units, hidden_units),
            nn.ELU(),
            nn.Linear(hidden_units, 1), 
            nn.Sigmoid()
        )


    def forward(self, covariates, **kwargs):
        #pad to ull seq length if needed

        phi = self.shared_base(covariates)
        outs = []
        for t in range(self.num_treatments):
            outs.append(getattr(self, f'head{t}')(phi))
        
        t_prob = self.propensity_head(phi)
        
        return t_prob, outs


    def predict_treatment_effect(self, treatments, covariates, **kwargs):
        _, ites = self(covariates, **kwargs)

        pred_stack = torch.stack(ites, dim=-1).squeeze()
        factual_predictions = torch.gather(pred_stack, 2, treatments.argmax(dim=-1).unsqueeze(-1)).squeeze()

        return factual_predictions
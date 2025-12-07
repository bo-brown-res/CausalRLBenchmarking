import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# 1. Gradient Reversal Layer (The "Adversarial" Magic)
class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, torch.tensor(alpha))
        return x

    @staticmethod
    def backward(ctx, grad_output):
        output, alpha = ctx.saved_tensors
        # Flip the gradient by multiplying by negative alpha
        return grad_output.neg() * alpha, None

class CRN(nn.Module):
    def __init__(self, num_covariates, num_treatments, num_outputs, 
                 rnn_hidden_units=64, fc_hidden_units=32):
        """
        Args:
            num_covariates: Dimension of patient features.
            num_treatments: Dimension of treatment (action) space.
            num_outputs: Dimension of the outcome (e.g., tumor size).
        """
        super(CRN, self).__init__()
        
        self.num_treatments = num_treatments
        
        # LSTM input = covariates + previous treatments
        self.lstm = nn.LSTM(input_size=num_covariates + num_treatments, 
                            hidden_size=rnn_hidden_units, 
                            batch_first=True)
        
        # -- Branch 1: Outcome Prediction (The Standard Task) --
        # Input: LSTM Representation + Current Treatment
        self.outcome_net = nn.Sequential(
            nn.Linear(rnn_hidden_units + num_treatments, fc_hidden_units),
            nn.ELU(),
            nn.Linear(fc_hidden_units, num_outputs)
        )
        
        # -- Branch 2: Treatment Prediction (The Adversarial Task) --
        # Input: LSTM Representation only
        # We want to FOOL this network, so we use Gradient Reversal
        self.treatment_net = nn.Sequential(
            nn.Linear(rnn_hidden_units, fc_hidden_units),
            nn.ELU(),
            nn.Linear(fc_hidden_units, num_treatments)
        )

    def forward(self, covariates, prev_treatments, current_treatments, alpha=1.0):
        """
        Args:
            covariates: (Batch, Seq_Len, Dim_Cov)
            prev_treatments: (Batch, Seq_Len, Dim_Treat) -> Actions taken at t-1
            current_treatments: (Batch, Seq_Len, Dim_Treat) -> Actions taken at t
            alpha: Strength of the gradient reversal (lambda in the paper)
        """
        
        # 1. Build Representation (History)
        # Concatenate X_t and A_{t-1}
        rnn_input = torch.cat([covariates, prev_treatments], dim=-1)
        
        # Run LSTM
        # rnn_output shape: (Batch, Seq_Len, Hidden_Units)
        rnn_output, _ = self.lstm(rnn_input)
        
        # 2. Outcome Prediction (Factual Prediction)
        # We assume the outcome Y_{t+1} depends on History (rnn_output) AND Current Action (A_t)
        outcome_input = torch.cat([rnn_output, current_treatments], dim=-1)
        outcome_pred = self.outcome_net(outcome_input)
        
        # 3. Treatment Prediction (Propensity Balancing)
        # Apply Gradient Reversal to the representation before this head
        # If we optimize to minimize treatment loss, the encoder will actually 
        # try to MAXIMIZE it (learn representations that hide treatment info).
        reversed_rnn_output = GradientReversal.apply(rnn_output, alpha)
        treatment_pred_logits = self.treatment_net(reversed_rnn_output)
        
        return outcome_pred, treatment_pred_logits
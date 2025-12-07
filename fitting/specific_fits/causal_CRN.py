import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence

def crn_forward(model, treatments, covariates, lambda_alpha, **kwargs):
    
    pad_packed_sequence(covariates, batch_first=True)

    batch_size, seq_len, treat_dim = treatments.data.shape
    zeros = torch.zeros(batch_size, 1, treat_dim).to(treatments.device)
    prev_treatments = torch.cat([zeros, treatments[:, :-1, :]], dim=1)

    pred_outcomes, pred_treatment_logits = model(
                covariates, 
                prev_treatments, 
                current_treatments=treatments,
                alpha=lambda_alpha
            )
    return pred_outcomes, pred_treatment_logits


def crn_loss(predictions, true_outcomes, treatments, lambda_alpha, criterion_outcome, criterion_treatment, **kwargs):
    batch_size, seq_len, treat_dim = treatments.shape
    pred_outcomes, pred_treatment_logits = predictions

    loss_outcome = criterion_outcome(pred_outcomes, true_outcomes)
    true_treatment_indices = torch.argmax(treatments, dim=-1).view(-1)
    pred_treatment_logits_flat = pred_treatment_logits.view(-1, treat_dim)
    
    loss_treatment = criterion_treatment(pred_treatment_logits_flat, true_treatment_indices)
    loss = loss_outcome + (lambda_alpha * loss_treatment)
    return loss



# def train_crn(model, dataloader, num_epochs=100, learning_rate=0.01, lambda_alpha=1.0):
#     """
#     Simple training loop for CRN.
#     """
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
#     # Loss functions
#     criterion_outcome = nn.MSELoss() # Regression for continuous outcomes (e.g. tumor size)
#     criterion_treatment = nn.CrossEntropyLoss() # Classification for treatments
    
#     model.train()
    
#     for epoch in range(num_epochs):
#         total_loss = 0
        
#         for batch in dataloader:
#             # Unpack batch (Assumes dataloader returns this tuple)
#             # covs: (B, T, D_cov), treats: (B, T, D_act), outputs: (B, T, D_out)
#             covariates, treatments, true_outcomes = batch
            
#             # Prepare inputs
#             # LSTM needs Previous treatments (Shift right by 1, pad with zeros at start)
#             batch_size, seq_len, treat_dim = treatments.shape
#             zeros = torch.zeros(batch_size, 1, treat_dim).to(treatments.device)
#             prev_treatments = torch.cat([zeros, treatments[:, :-1, :]], dim=1)
            
#             # Forward pass
#             pred_outcomes, pred_treatment_logits = model(
#                 covariates, 
#                 prev_treatments, 
#                 current_treatments=treatments,
#                 alpha=lambda_alpha
#             )
            
#             # --- Calculate Losses ---
            
#             # 1. Outcome Loss (MSE)
#             loss_outcome = criterion_outcome(pred_outcomes, true_outcomes)
            
#             # 2. Treatment Loss (CrossEntropy)
#             # Flatten Sequence dimensions for CrossEntropy: (B*T, Classes)
#             # Assuming treatments are one-hot encoded in input, we need indices for CrossEntropy
#             true_treatment_indices = torch.argmax(treatments, dim=-1).view(-1)
#             pred_treatment_logits_flat = pred_treatment_logits.view(-1, treat_dim)
            
#             loss_treatment = criterion_treatment(pred_treatment_logits_flat, true_treatment_indices)
            
#             # 3. Total Loss (Adversarial Balance)
#             # Because of the Gradient Reversal Layer inside the model, 
#             # minimizing 'loss_treatment' here effectively MAXIMIZES the confusion 
#             # of the encoder regarding treatments.
#             loss = loss_outcome + (lambda_alpha * loss_treatment)
            
#             # Optimization
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             total_loss += loss.item()
            
#         print(f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss:.4f}")

#     return model


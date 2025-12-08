import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence



def crn_loss(predictions, true_outcomes, treatments, lambda_alpha, criterion_outcome, criterion_treatment, **kwargs):
    batch_size, seq_len, treat_dim = treatments.shape
    pred_outcomes, pred_treatment_logits = predictions

    loss_outcome = criterion_outcome(pred_outcomes, true_outcomes)
    true_treatment_indices = torch.argmax(treatments, dim=-1).view(-1)
    pred_treatment_logits_flat = pred_treatment_logits.view(-1, treat_dim)
    
    loss_treatment = criterion_treatment(pred_treatment_logits_flat, true_treatment_indices)
    loss = loss_outcome + (lambda_alpha * loss_treatment)
    return loss



def tarnet_loss(predictions, true_outcomes, treatments, **kwargs):
    # 1. Stack the list of predictions into a single tensor
    # Shape becomes: (Batch, Num_Treatments, Output_Dim)
    pred_stack = torch.stack(predictions, dim=1)
    
    # 2. Prepare treatments for gathering
    # We need to expand treatments to match the Output_Dim if necessary
    # Shape changes from (Batch,) -> (Batch, 1, Output_Dim) for gathering
    t_expanded = treatments.view(-1, 1, 1).expand(-1, 1, true_outcomes.shape[-1])
    
    # 3. Select the prediction corresponding to the actual treatment
    # We gather along dim=1 (the treatment dimension)
    factual_predictions = torch.gather(pred_stack, 1, t_expanded).squeeze(1)
    
    # 4. Compute standard loss (MSE for regression, CrossEntropy for classification)
    # The paper typically uses MSE for continuous outcomes
    loss = torch.nn.functional.mse_loss(factual_predictions, true_outcomes)
    
    return loss


def dragonnet_loss(predictions, true_outcomes, treatments, dragon_alpha, **kwargs):
    propensity_logits, pred_logits = predictions
    pred_stack = torch.stack(pred_logits, dim=-1).squeeze()
    
    # Expand treatments for gathering: (Batch, 1, Output_Dim)
    # t_expanded = treatments.view(-1, 1, 1).expand(-1, 1, true_outcomes.shape[-1])
    
    # Gather factual predictions
    factual_predictions = torch.gather(pred_stack, 2, treatments.argmax(dim=-1).unsqueeze(-1)).squeeze()
    
    # Standard Regression Loss (MSE)
    loss_outcome = torch.nn.functional.mse_loss(factual_predictions, true_outcomes.squeeze())
    
    # --- 2. Propensity Loss ---
    # We use BCEWithLogitsLoss for numerical stability
    # treatments must be float for BCE, shape (Batch, 1)
    t_float = treatments.view(-1, 1).float()
    loss_propensity = torch.nn.functional.binary_cross_entropy_with_logits(
        propensity_logits.squeeze(), 
        treatments.argmax(dim=-1).float())
    
    # --- 3. Total Loss ---
    total_loss = loss_outcome + (dragon_alpha * loss_propensity)
    
    return total_loss
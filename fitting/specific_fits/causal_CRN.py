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



def tarnet_loss(predictions, true_outcomes, treatments, targets='return', **kwargs):
    pred_stack = torch.stack(predictions, dim=-1).squeeze()
    factual_predictions = torch.gather(pred_stack, 2, treatments.argmax(dim=-1).unsqueeze(-1)).squeeze()
    
    if targets == 'return':
        factual_preds = factual_predictions.sum(-1)
    else:
        factual_preds = factual_predictions
    loss_outcome = torch.nn.functional.mse_loss(factual_preds, true_outcomes.squeeze())
    return loss_outcome


def dragonnet_loss(predictions, true_outcomes, treatments, dragon_alpha, **kwargs):
    propensity_logits, pred_logits = predictions
    pred_stack = torch.stack(pred_logits, dim=-1).squeeze()
    #map predicted treatments to actually selected treatments - onyl fitting on treatments actually seen in data as per paper
    factual_predictions = torch.gather(pred_stack, 2, treatments.argmax(dim=-1).unsqueeze(-1)).squeeze()
    loss_outcome = torch.nn.functional.mse_loss(factual_predictions, true_outcomes.squeeze())
    
    #compute the secondary propensity score prediction loss
    loss_propensity = torch.nn.functional.binary_cross_entropy_with_logits(
        propensity_logits.squeeze(), 
        treatments.argmax(dim=-1).float())
    
    total_loss = loss_outcome + (dragon_alpha * loss_propensity)
    return total_loss
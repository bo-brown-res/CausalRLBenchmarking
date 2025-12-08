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

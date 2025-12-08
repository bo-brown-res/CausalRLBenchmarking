import torch
import torch.nn.functional as F

def t4_loss_function(pred_outcomes, true_outcomes, pred_propensity, true_treatments, alpha=1.0):
    """
    Computes T4 loss: Outcome MSE + Propensity BCE.
    
    Args:
        pred_outcomes: (Batch, Future_Len, 1)
        true_outcomes: (Batch, Future_Len, 1)
        pred_propensity: (Batch, Hist_Len, 1)
        true_treatments: (Batch, Hist_Len, 1)
        alpha: Weight for propensity loss (usually used during pre-training)
    """
    # 1. Outcome Loss (MSE) - Eq (17) in paper
    outcome_loss = F.mse_loss(pred_outcomes, true_outcomes)
    
    # 2. Propensity Loss (BCE With Logits)
    # Used for the pre-training phase or auxiliary task
    propensity_loss = F.binary_cross_entropy_with_logits(pred_propensity, true_treatments)
    
    return outcome_loss + alpha * propensity_loss


def update_global_propensities(model, full_dataset_loader, device='cpu'):
    """
    Computes propensity scores for the entire training set.
    Run this at the beginning of every epoch to refresh the matching logic.
    
    Returns:
        all_propensities: (Total_Samples, 1) tensor
    """
    model.eval()
    all_props = []
    
    with torch.no_grad():
        for batch in full_dataset_loader:
            cov, static, treat, _ = batch # Assuming 4 items in dataset
            cov = cov.to(device)
            static = static.to(device)
            treat = treat.to(device)
            
            # Forward pass to get propensity logits
            # future_treatments is irrelevant for propensity, pass dummy
            dummy_future = torch.zeros(cov.size(0), 1, 1).to(device)
            
            _, prop_logits = model(cov, static, treat, dummy_future)
            
            # Store scores (sigmoid applied)
            # We usually take the last time step's propensity or average
            # Here we take the last step: shape (Batch, 1)
            all_props.append(torch.sigmoid(prop_logits[:, -1, :]).cpu())
            
    return torch.cat(all_props, dim=0).to(device)



def get_balanced_batch_global(model, batch_data, full_dataset_tensors, full_propensities, device='cpu'):
    """
    Finds the best counterfactual match for the mini-batch from the FULL dataset.
    
    Args:
        model: The T4 model.
        batch_data: Tuple (cov, static, treat, out) for the current mini-batch.
        full_dataset_tensors: Tuple (all_cov, all_static, all_treat, all_out) containing the WHOLE training set.
                              Must be on the same device as batch_data for speed.
        full_propensities: (N, 1) Pre-computed propensities for the full dataset.
    
    Returns:
        balanced_cov, balanced_static, ... (Combined batch of size 2 * Batch_Size)
    """
    model.eval()
    
    # Unpack current batch
    b_cov, b_static, b_treat, b_out = batch_data
    batch_size = b_cov.size(0)
    
    # Unpack full dataset
    all_cov, all_static, all_treat, all_out = full_dataset_tensors
    
    with torch.no_grad():
        # 1. Get Propensity for the CURRENT mini-batch
        # We compute this fresh because the batch is already loaded
        dummy_future = torch.zeros(batch_size, 1, 1).to(device)
        _, b_logits = model(b_cov, b_static, b_treat, dummy_future)
        b_props = torch.sigmoid(b_logits[:, -1, :]) # (Batch_Size, 1)
        
        # 2. Compute Distance Matrix (Vectorized)
        # Shape: (Batch_Size, Total_Dataset_Size)
        # abs(Batch_Prop - Full_Prop_Transposed)
        dists = torch.abs(b_props - full_propensities.T)
        
        # 3. Mask out same treatments
        # We strictly want T=0 matched with T=1, and vice versa.
        # Check where batch treatment equals dataset treatment
        # b_treat: (B, 1), all_treat: (N, 1) -> Broadcast to (B, N)
        # Note: We look at the last time step of treatment (or aggregated status)
        b_t_status = b_treat[:, -1, :] 
        all_t_status = all_treat[:, -1, :]
        
        same_treatment_mask = (b_t_status == all_t_status.T)
        
        # Set distance to infinity where treatments are the same (so we don't select them)
        dists[same_treatment_mask] = float('inf')
        
        # 4. Find nearest neighbor indices
        # min over dim=1 returns values, indices
        _, best_indices = torch.min(dists, dim=1) # (Batch_Size,)
        
    # 5. Gather the matched data from the full dataset
    matched_cov = all_cov[best_indices]
    matched_static = all_static[best_indices]
    matched_treat = all_treat[best_indices]
    matched_out = all_out[best_indices]
    
    # 6. Concatenate
    balanced_cov = torch.cat([b_cov, matched_cov], dim=0)
    balanced_static = torch.cat([b_static, matched_static], dim=0)
    balanced_treat = torch.cat([b_treat, matched_treat], dim=0)
    balanced_out = torch.cat([b_out, matched_out], dim=0)
    
    return balanced_cov, balanced_static, balanced_treat, balanced_out
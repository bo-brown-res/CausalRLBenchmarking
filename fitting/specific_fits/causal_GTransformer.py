import torch


def train_g_transformer_iteratively(model, dataloader, optimizer, num_epochs, device):
    """
    Implements the custom 'Iterative Training' from Algorithm 1/Appendix A.1.
    """
    mse_criterion = nn.MSELoss()
    ce_criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch in dataloader:
            # batch: (Batch, Seq_Len, Dim)
            # Unpack: Treats, Cats, Conts
            treats, cats, conts = batch
            treats, cats, conts = treats.to(device), cats.to(device), conts.to(device)
            
            batch_loss = 0
            seq_len = treats.size(1)
            
            # --- Iterative Training Loop ---
            # Instead of predicting the whole sequence at once, we loop through time t
            # and treat 0...t as the input to predict t+1.
            for t in range(1, seq_len - 1):
                
                # 1. Slice History (0 to t)
                curr_treats = treats[:, :t, :]
                curr_covs = torch.cat([cats[:, :t, :], conts[:, :t, :]], dim=-1)
                
                # 2. Target for t+1
                target_cat = cats[:, t, :] # (Batch, Dim_Cat)
                target_cont = conts[:, t, :] # (Batch, Dim_Cont)
                
                # 3. Forward Pass
                # Teacher Forcing: We pass the TRUE categorical info for t+1 to the continuous encoder
                # Note: We need to shape target_cat as (Batch, 1, Dim) for broadcasting in forward
                next_cat_input = target_cat.unsqueeze(1)
                
                pred_cat_logits, pred_cont = model(curr_treats, curr_covs, next_cat_input)
                
                # 4. Calculate Loss
                # Continuous Loss [cite: 1679]
                loss_cont = mse_criterion(pred_cont, target_cont)
                
                # Categorical Loss [cite: 1673]
                loss_cat = 0
                # Assuming simple binary/one-hot. If multiple cat vars, loop and sum CE.
                # Here assuming 1 cat var for simplicity
                # pred_cat_logits[0]: (Batch, Num_Classes)
                # target_cat: (Batch, 1) or One-Hot
                target_cat_idx = torch.argmax(target_cat, dim=-1)
                loss_cat += ce_criterion(pred_cat_logits[0], target_cat_idx)
                
                step_loss = loss_cont + loss_cat
                batch_loss += step_loss
            
            # Optimize over the accumulated sequence loss
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            total_loss += batch_loss.item()
            
        print(f"Epoch {epoch}: Loss {total_loss}")


def simulate_counterfactual(model, history_treats, history_covs, treatment_policy_fn, future_steps):
    """
    Performs Monte Carlo simulation (G-Computation) for future steps.
    """
    model.eval()
    current_treats = history_treats
    current_covs = history_covs
    
    predictions = []
    
    with torch.no_grad():
        for _ in range(future_steps):
            # 1. Determine NEXT Action using the Policy Function (g)
            # The policy looks at the full history up to now
            # g(H_t) -> A_t
            next_action = treatment_policy_fn(current_covs, current_treats)
            
            # Append action to history
            current_treats = torch.cat([current_treats, next_action.unsqueeze(1)], dim=1)
            
            # 2. Predict NEXT Categorical Covariates (L_ca)
            # Input: History up to t (including the action we just decided)
            # We assume current_covs is length T, current_treats is length T+1
            # We slice treats to match covs for the 'prev' argument
            
            pred_logits, _ = model(current_treats[:, :-1, :], current_covs, current_cat_covariates=None)
            
            # Sample from logits (Monte Carlo)
            # For simplicity, we take argmax here (point estimate)
            # In full g-comp, you would sample from the distribution
            pred_cat = torch.argmax(pred_logits[0], dim=-1).unsqueeze(-1).float() # (Batch, 1)
            
            # 3. Predict NEXT Continuous Covariates (L_co)
            # Use the predicted cat as input
            _, pred_cont = model(current_treats[:, :-1, :], current_covs, pred_cat.unsqueeze(1))
            
            # 4. Update History
            # Combine Cat and Cont
            next_cov = torch.cat([pred_cat, pred_cont], dim=-1).unsqueeze(1)
            current_covs = torch.cat([current_covs, next_cov], dim=1)
            
            predictions.append(next_cov)
            
    return torch.cat(predictions, dim=1)
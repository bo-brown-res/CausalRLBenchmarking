# https://arxiv.org/abs/2406.05504


import torch
import torch.nn as nn
import torch.nn.functional as F

class GTransformer(nn.Module):
    def __init__(self, num_continuous, num_categorical, num_treatments, 
                 num_cat_classes, d_model=64, nhead=2, num_layers=2, dropout=0.1):
        """
        G-Transformer for Counterfactual Outcome Prediction.
        
        Args:
            num_continuous: Number of continuous covariates.
            num_categorical: Number of categorical covariates.
            num_treatments: Dimension of treatment vector.
            num_cat_classes: List containing number of classes for each categorical var.
            d_model: Hidden dimension for Transformer.
        """
        super(GTransformer, self).__init__()
        
        self.d_model = d_model
        
        # --- 1. Embeddings ---
        # Input to Cat Encoder: [History_Treat, History_Cat, History_Cont]
        # We project the concatenated raw input to d_model size
        input_size = num_treatments + num_categorical + num_continuous
        self.cat_input_proj = nn.Linear(input_size, d_model)
        
        # Input to Cont Encoder: [History_Treat, History_Cat, History_Cont, Current_Cat]
        # Continuous encoder sees the 'current' categorical outcome to predict continuous
        cont_input_size = input_size + num_categorical 
        self.cont_input_proj = nn.Linear(cont_input_size, d_model)

        # --- 2. Transformer Encoders ---
        # "We utilized two Transformer encoders... to separately learn hidden representations" 
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=d_model*4, 
                                                   dropout=dropout, batch_first=True)
        
        self.cat_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cont_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # --- 3. Prediction Heads ---
        # Categorical Heads (One per categorical variable)
        self.cat_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, num_classes)
            ) for num_classes in num_cat_classes
        ])
        
        # Continuous Head (Predicts all continuous variables at once)
        self.cont_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_continuous)
        )

    def forward(self, prev_treatments, prev_covariates, current_cat_covariates=None):
        """
        Forward pass for a single time step t prediction.
        
        Args:
            prev_treatments: (Batch, Seq_Len, Dim_Treat)
            prev_covariates: (Batch, Seq_Len, Dim_Cov) - Concat of [Cat, Cont]
            current_cat_covariates: (Batch, 1, Dim_Cat) - Known/Predicted cat vars at time t
                                    Required for predicting continuous vars.
        
        Returns:
            pred_cat_logits: List of Tensors for each categorical variable
            pred_cont: Tensor (Batch, 1, Dim_Cont)
        """
        # 1. Categorical Prediction Branch
        # Concatenate history
        hist_input = torch.cat([prev_treatments, prev_covariates], dim=-1) # (B, S, Input)
        
        # Project and Encode
        cat_embed = self.cat_input_proj(hist_input) # Eq (6) [cite: 1645]
        
        # Causal Mask (Optional for Encoder if training step-by-step, but standard for sequence)
        # For iterative training, we usually just pass the valid history sequence.
        h_ca = self.cat_encoder(cat_embed) # Eq (7) [cite: 1648]
        
        # We take the LAST hidden state to predict the NEXT time step
        last_h_ca = h_ca[:, -1, :] 
        
        pred_cat_logits = [head(last_h_ca) for head in self.cat_heads] # Eq (8) [cite: 1653]
        
        # 2. Continuous Prediction Branch
        # If training, we use Ground Truth current_cat. If simulating, we use Softmax(pred_cat).
        # We need to construct the input for the continuous encoder: [History, Current_Cat]
        
        # We expect current_cat_covariates to be provided (Teacher Forcing) or sampled
        if current_cat_covariates is None:
            # Greedy decoding for inference if not provided
            # Convert logits to one-hot or similar embedding
            # simplified: stacking argmax
            pred_cats = [torch.argmax(logits, dim=-1).unsqueeze(-1).float() for logits in pred_cat_logits]
            current_cat_covariates = torch.cat(pred_cats, dim=-1).unsqueeze(1)
            
        # For the continuous encoder, we align the sequences.
        # The paper says: input = concat(A_t, L_t, L_{t+1}^ca) [cite: 1657]
        # We must align the sequence length. 
        # Typically G-Transformer treats the sequence up to t-1 + current t prediction
        
        # Expand current_cat to match sequence history length for simple concatenation 
        # OR (more likely) we append the new token to the sequence.
        # Based on Eq 9, it seems we augment the current step input.
        
        # Re-construct history with the "current" categorical info baked in?
        # Actually, the paper implies we use the same history + the new cat info.
        # Let's append the current cat info to the END of the history for the last step
        
        # Simplified: Continuous encoder takes history + current categorical prediction
        # To make dimensions match, we might repeat the cat vector or concat along feature dim.
        # Here we concat along feature dim, assuming cat vars are available for all history steps.
        
        # Note: In training, 'prev_covariates' already contains cat vars for 0..t-1.
        # We need cat var for t.
        
        # Input: [History_Treat, History_Cat, History_Cont] + [Zero, Current_Cat, Zero] ?
        # Efficient Implementation: 
        # We reuse the history embedding and just add the new information for the last step.
        
        # Let's stick to the equation: h_co input includes L_{t+1}^ca
        # We concat current cat to the history features.
        # To do this cleanly, we assume the input sequence includes the current step for Cont Encoder
        
        cont_embed = self.cont_input_proj(
            torch.cat([hist_input, current_cat_covariates.repeat(1, hist_input.size(1), 1)], dim=-1) 
        ) # Eq (9) [cite: 1657]
        
        h_co = self.cont_encoder(cont_embed) # Eq (10) [cite: 1661]
        last_h_co = h_co[:, -1, :]
        
        pred_cont = self.cont_head(last_h_co) # Eq (11) [cite: 1661]
        
        return pred_cat_logits, pred_cont
    

    def simulate_trajectory(self, history_treats, history_covs, forced_action, future_steps=1):
        """
        Helper: Simulates the future for N steps given a specific STARTING action.
        Subsequent actions are assumed to be 0 (or you can add a policy policy arg).
        """
        curr_treats = history_treats.clone()
        curr_covs = history_covs.clone()
        
        # Step 1: Force the specific treatment action for the immediate next step
        action_t = forced_action.unsqueeze(1) # (Batch, 1, Dim_Treat)
        curr_treats = torch.cat([curr_treats, action_t], dim=1)
        
        # Predict t+1 Covariates
        logits, _ = self(curr_treats[:, :-1, :], curr_covs, current_cat_covariates=None)
        pred_cat = torch.argmax(logits[0], dim=-1).unsqueeze(-1).float() 
        _, pred_cont = self(curr_treats[:, :-1, :], curr_covs, pred_cat.unsqueeze(1))
        
        next_cov = torch.cat([pred_cat, pred_cont], dim=-1).unsqueeze(1)
        curr_covs = torch.cat([curr_covs, next_cov], dim=1)
        
        # Step 2...N: Simulate further (assuming no further treatment / standard of care)
        # Note: In pure ITE, we usually only care about the immediate outcome or cumulative outcome
        for _ in range(future_steps - 1):
            # Assume no treatment (zeros) for future steps, or repeat last action
            future_action = torch.zeros_like(action_t) 
            curr_treats = torch.cat([curr_treats, future_action], dim=1)
            
            logits, _ = self(curr_treats[:, :-1, :], curr_covs, current_cat_covariates=None)
            pred_cat = torch.argmax(logits[0], dim=-1).unsqueeze(-1).float()
            _, pred_cont = self(curr_treats[:, :-1, :], curr_covs, pred_cat.unsqueeze(1))
            
            next_cov = torch.cat([pred_cat, pred_cont], dim=-1).unsqueeze(1)
            curr_covs = torch.cat([curr_covs, next_cov], dim=1)
            
        return curr_covs # Return the full trajectory (or just the final state)

    def predict_treatment_effect(self, history_treats, history_covs, target_treatment, baseline_treatment=None, future_steps=1):
        """
        Predicts the Individual Treatment Effect (ITE) by simulating two parallel universes.
        
        Args:
            history_treats: (Batch, Seq_Len, Dim_Treat)
            history_covs: (Batch, Seq_Len, Dim_Cov)
            target_treatment: (Batch, Dim_Treat) - The action to test at time t
            baseline_treatment: (Batch, Dim_Treat) - Reference action (default: zeros)
            future_steps: How many steps into the future to simulate.
            
        Returns:
            ite: (Batch, Future_Steps, Dim_Cov) or just (Batch, Dim_Cov)
        """
        was_training = self.training
        self.eval()
        
        with torch.no_grad():
            if baseline_treatment is None:
                baseline_treatment = torch.zeros_like(target_treatment)
            
            # 1. Simulate "Treated" World
            # Returns shape: (Batch, Hist+Future, Dim_Cov)
            traj_treated = self.simulate_trajectory(
                history_treats, history_covs, target_treatment, future_steps
            )
            
            # 2. Simulate "Control" World
            traj_control = self.simulate_trajectory(
                history_treats, history_covs, baseline_treatment, future_steps
            )
            
            # 3. Calculate Effect
            # We usually only care about the *newly predicted* steps, not the history
            # Extract the last 'future_steps'
            outcome_treated = traj_treated[:, -future_steps:, :]
            outcome_control = traj_control[:, -future_steps:, :]
            
            ite = outcome_treated - outcome_control
            
        self.train(mode=was_training)
        
        # If steps=1, squeeze the sequence dim for convenience
        if future_steps == 1:
            return ite.squeeze(1)
            
        return ite
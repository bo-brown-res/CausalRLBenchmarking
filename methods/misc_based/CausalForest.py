import torch
import numpy as np
from econml.grf import CausalForest 

class CausalForestModel:
    def __init__(self, n_estimators=200, min_samples_leaf=10, max_depth=None, **kwargs):
        """
        A wrapper around the CausalForest algorithm (GRF) that handles 
        PyTorch sequence tensors.
        """
        self.forest = CausalForest(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            criterion='het', 
            honest=True,      
            **kwargs
        )
        self.is_fitted = False

    def fit(self, covariates, treatments, outcomes):
        """
        Fits the Forest on the provided batch of sequences.
        """
        # 1. Convert PyTorch Tensors to Numpy
        if isinstance(covariates, torch.Tensor):
            covariates = covariates.cpu().detach().numpy()
        if isinstance(treatments, torch.Tensor):
            treatments = treatments.cpu().detach().numpy()
        if isinstance(outcomes, torch.Tensor):
            outcomes = outcomes.cpu().detach().numpy()
            
        # 2. Flatten the Sequence Dimension
        # The Forest treats every timestep as an independent sample.
        X = covariates.reshape(-1, covariates.shape[-1])
        
        # Handle Treatment Flattening
        if treatments.ndim == 3:
            T = treatments.reshape(-1, treatments.shape[-1])
        else:
            T = treatments.flatten()
            
        # Handle Outcome Flattening
        if outcomes.ndim == 3:
            Y = outcomes.reshape(-1, outcomes.shape[-1])
        else:
            Y = outcomes.flatten()
            
        # 3. Fit the Causal Forest
        self.forest.fit(X=X, T=T, y=Y)
        self.is_fitted = True
        
        return self

    def predict_treatment_effect(self, covariates, target_treatment, baseline_treatment=None):
        """
        Predicts the treatment effect of taking 'target_treatment' vs 'baseline_treatment'.
        
        Formula: Effect = (Target - Baseline) * Tau(Covariates)
        
        Args:
            covariates: (Batch, Seq_Len, Dim_Cov) - The State
            target_treatment: (Batch, Seq_Len, Dim_Treat) - The Action to take
            baseline_treatment: (Batch, Seq_Len, Dim_Treat) - Optional reference (default 0)
            
        Returns:
            effect_tensor: (Batch, Seq_Len, Output_Dim)
        """
        if not self.is_fitted:
            raise RuntimeError("You must call .fit() before predicting.")
            
        # 1. Capture Shapes
        batch_size, seq_len, dim_cov = covariates.shape
        
        # 2. Prepare Covariates (State)
        if isinstance(covariates, torch.Tensor):
            X_flat = covariates.cpu().detach().numpy().reshape(-1, dim_cov)
        else:
            X_flat = covariates.reshape(-1, dim_cov)

        # 3. Predict Tau (Heterogeneous Treatment Effect)
        # Tau represents the effect of a 1-unit increase in treatment
        # shape: (Batch * Seq_Len, Output_Dim)
        tau_hat_flat = self.forest.predict(X_flat)
        
        # Ensure tau is 2D for broadcasting: (N, Out_Dim)
        if tau_hat_flat.ndim == 1:
            tau_hat_flat = tau_hat_flat[:, None]

        # 4. Prepare Treatments (Actions)
        # We need to handle potential scalar inputs or full tensors
        if baseline_treatment is None:
            baseline_treatment = torch.zeros_like(target_treatment)
            
        # Convert to numpy for calculation
        if isinstance(target_treatment, torch.Tensor):
            t_target = target_treatment.cpu().detach().numpy()
        else:
            t_target = target_treatment
            
        if isinstance(baseline_treatment, torch.Tensor):
            t_base = baseline_treatment.cpu().detach().numpy()
        else:
            t_base = baseline_treatment
            
        # Flatten treatments to match X_flat
        # Target shape becomes (Batch * Seq_Len, Dim_Treat)
        t_target_flat = t_target.reshape(-1, t_target.shape[-1] if t_target.ndim > 2 else 1)
        t_base_flat = t_base.reshape(-1, t_base.shape[-1] if t_base.ndim > 2 else 1)

        # 5. Calculate Specific Effect
        # Logic: If Tau is "Effect of Unit Change", then total effect is (Delta Action) * Tau
        delta_action = t_target_flat - t_base_flat
        
        # Multiply: (N, Treat_Dim) * (N, Out_Dim) -> (N, Out_Dim)
        # Note: This assumes Treat_Dim == 1 (standard for GRF) or element-wise broadcasting
        # If you have multiple treatments and multiple outcomes, EconML output shapes vary.
        # Here we assume standard single-treatment setup.
        effect_flat = delta_action * tau_hat_flat
        
        # 6. Reshape back to Sequence
        output_dim = effect_flat.shape[-1]
        effect_seq = effect_flat.reshape(batch_size, seq_len, output_dim)
        
        ite = torch.tensor(effect_seq, dtype=torch.float32)
        return ite
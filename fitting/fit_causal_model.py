import torch
import torch.nn as nn


# if __name__ == "__main__":
#     # Hyperparameters
#     BATCH_SIZE = 32
#     SEQ_LEN = 10
#     NUM_COVARIATES = 5
#     NUM_TREATMENTS = 2  # e.g., Treat A or Treat B
#     NUM_OUTPUTS = 1     # e.g., Tumor volume
#     # Create the model
#     crn = CRN(NUM_COVARIATES, NUM_TREATMENTS, NUM_OUTPUTS)
#     # Generate Dummy Data
#     # In a real scenario, use your own Dataset/DataLoader
#     mock_covariates = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_COVARIATES)
#     # One-hot treatments
#     mock_treatments_idx = torch.randint(0, NUM_TREATMENTS, (BATCH_SIZE, SEQ_LEN))
#     mock_treatments = F.one_hot(mock_treatments_idx, NUM_TREATMENTS).float()
#     mock_outcomes = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_OUTPUTS)
#     # Wrap in a simple list for the loop
#     mock_dataloader = [(mock_covariates, mock_treatments, mock_outcomes)]
#     # Train
#     trained_model = train_crn(crn, mock_dataloader, num_epochs=5)


def setup_and_run_cs(
            method_name,
            train_config,
            train_dataset,
            val_dataset,
            test_dataset,
        ):

    if method_name == 'CRN':
        from methods.causal_based.CounterfactualRecurrentNetwork import CRN
        from fitting.specific_fits.causal_CRN import crn_forward, crn_loss
        model = CRN(
            num_covariates=train_dataset.dataset.data[0].shape[-1],
            num_treatments=train_dataset.dataset.data[1].shape[-1],
            num_outputs=train_dataset.dataset.data[2].shape[-1],
        )
        model_forward_fn = crn_forward
        model_loss_fn = crn_loss

        kwargs = {
            'lambda_alpha': train_config.get('lambda_alpha', 1.0),
            'criterion_outcome': nn.MSELoss(),
            'criterion_treatment': nn.CrossEntropyLoss(),
        }

    else:
        raise ValueError(f"Unknown causal method name: {method_name}")

    trained_model = training_loop(
        model,
        train_dataset,
        model_forward_fn,
        model_loss_fn,
        num_epochs=train_config.get('n_steps', 100),
        learning_rate=train_config.get('learning_rate', 0.001),
        **kwargs
    )

    return trained_model



def training_loop(model, dataloader, model_forward_fn, model_loss_fn, num_epochs=100, learning_rate=0.01, **kwargs):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch in dataloader :
            covariates, treatments, rewards, terminations = batch
            
            predictions = model_forward_fn(
                model=model, 
                treatments=treatments,
                covariates=covariates, 
                # lambda_alpha=kwargs.get('lambda_alpha', 1.0)
                **kwargs
            )

            loss = model_loss_fn(predictions, rewards, treatments, **kwargs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss:.4f}")

    return model
import torch
import torch.nn as nn


if __name__ == "__main__":
    # Hyperparameters
    BATCH_SIZE = 32
    SEQ_LEN = 10
    NUM_COVARIATES = 5
    NUM_TREATMENTS = 2  # e.g., Treat A or Treat B
    NUM_OUTPUTS = 1     # e.g., Tumor volume
    
    # Create the model
    crn = CRN(NUM_COVARIATES, NUM_TREATMENTS, NUM_OUTPUTS)
    
    # Generate Dummy Data
    # In a real scenario, use your own Dataset/DataLoader
    mock_covariates = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_COVARIATES)
    
    # One-hot treatments
    mock_treatments_idx = torch.randint(0, NUM_TREATMENTS, (BATCH_SIZE, SEQ_LEN))
    mock_treatments = F.one_hot(mock_treatments_idx, NUM_TREATMENTS).float()
    
    mock_outcomes = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_OUTPUTS)
    
    # Wrap in a simple list for the loop
    mock_dataloader = [(mock_covariates, mock_treatments, mock_outcomes)]
    
    # Train
    trained_model = train_crn(crn, mock_dataloader, num_epochs=5)


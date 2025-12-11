import csv
import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd

from evaluate.action_val_eval import compute_true_ite_error



def setup_and_run_cs(
            method_name,
            train_config,
            train_dataset,
            val_dataset,
            test_dataset,
            dataset_name,
        ):

    kwargs = {
        'device': train_config.get('device', 'cpu'),
        'ds_name': dataset_name
    }

    if method_name == 'CRN':
        from methods.causal_based.CounterfactualRecurrentNetwork import CRN
        from fitting.specific_fits.causal_CRN import crn_loss
        model = CRN(
            num_covariates=train_dataset.dataset[0][0].shape[-1],
            num_treatments=train_dataset.dataset[0][1].shape[-1],
            num_outputs=train_dataset.dataset[0][2].shape[-1],
            rnn_hidden_units=train_config.get('rnn_hidden_units', 64),
            fc_hidden_units=train_config.get('fc_hidden_units', 32),
        ).to(train_config.get('device'))

        model_loss_fn = crn_loss

        kwargs.update({
            'lambda_alpha': train_config.get('lambda_alpha', 1.0),
            'criterion_outcome': nn.MSELoss(),
            'criterion_treatment': nn.CrossEntropyLoss(),
        })
    elif method_name == 'TARNet':
        from methods.causal_based.TARNet import TARNet
        from fitting.specific_fits.causal_CRN import tarnet_loss

        model = TARNet(
            num_covariates=train_dataset.dataset[0][0].shape[-1],
            num_treatments=train_dataset.dataset[0][1].shape[-1],
            num_outputs=train_dataset.dataset[0][2].shape[-1],
            hidden_units=train_config.get('hidden_units', 64),
        ).to(train_config.get('device'))

        model_loss_fn = tarnet_loss

        kwargs.update({
            'hidden_units': train_config.get('hidden_units', 64),
        })
    elif method_name == 'DragonNet':
        from methods.causal_based.DragonNet import DragonNet
        from fitting.specific_fits.causal_CRN import dragonnet_loss

        model = DragonNet(
            num_covariates=train_dataset.dataset[0][0].shape[-1],
            num_treatments=train_dataset.dataset[0][1].shape[-1],
            num_outputs=train_dataset.dataset[0][2].shape[-1],
            hidden_units=train_config.get('hidden_units', 64),
        ).to(train_config.get('device'))

        model_loss_fn = dragonnet_loss

        kwargs.update({
            'hidden_units': train_config.get('hidden_units', 64),
            'dragon_alpha': train_config.get('dragon_alpha', 1.0),
        })
    else:
        raise ValueError(f"Unknown causal method name: {method_name}")
    

    my_metricsfn_dict = {
        'true_ite_error': compute_true_ite_error
    }

    trained_model, train_results = training_loop(
        model=model,
        train_dataloader=train_dataset,
        val_dataloader=val_dataset,
        model_forward_fn=model.forward,
        model_loss_fn=model_loss_fn,
        num_epochs=train_config.get('n_steps', 100),
        learning_rate=train_config.get('learning_rate', 0.001),
        metricsfn_dict=my_metricsfn_dict,
        **kwargs
    )

    test_results = test_model(
        model,
        test_dataloader=test_dataset,
        model_forward_fn=model.forward,
        model_loss_fn=model_loss_fn,
        metricsfn_dict=my_metricsfn_dict,
        **kwargs
    )

    return trained_model, train_results, test_results



def training_loop(model, train_dataloader, val_dataloader, model_forward_fn, model_loss_fn, 
                  num_epochs=100, learning_rate=0.01, csv_path="metrics.csv", earlystop_wait=10,
                  metricsfn_dict={}, **kwargs):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    device = kwargs.get('device', 'cpu')

    best_val_loss = float('inf')
    epochs_no_improve = 0

    records = []

    for epoch in range(num_epochs):
        model.train()
        train_loss_accum = 0.0
        train_batches = 0
        
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in pbar:
            covariates, treatments, rewards, terminations, true_ites, mask = batch
            covariates = covariates.to(device)
            treatments = treatments.to(device)
            rewards = rewards.to(device)
            terminations = terminations.to(device)
            
            predictions = model_forward_fn(
                model=model, 
                treatments=treatments,
                covariates=covariates, 
                **kwargs
            )

            loss = model_loss_fn(predictions, rewards, treatments, **kwargs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_accum += loss.item()
            train_batches += 1
            
            pbar.set_postfix({'train_loss': loss.item()})

        avg_train_loss = train_loss_accum / train_batches if train_batches > 0 else 0

        validation_results = evaluate_dataset(
            model=model, 
            dataloader=val_dataloader, 
            model_forward_fn=model_forward_fn, 
            model_loss_fn=model_loss_fn, 
            metricsfn_dict=metricsfn_dict, 
            **kwargs
        )

        val_loss = validation_results.get('val_avg_loss', float('inf'))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= earlystop_wait:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        records.append({**{
            'epoch': epoch+1,
            'avg_train_loss': avg_train_loss,
        }, **validation_results})

        train_results = pd.DataFrame(records)
        train_results.to_csv(f"training_{model.__class__.__name__}_{kwargs['ds_name']}.csv")
    return model, train_results

def test_model(model, test_dataloader, model_forward_fn, model_loss_fn, metricsfn_dict, **kwargs):
    validation_results = evaluate_dataset(
        model=model, 
        dataloader=test_dataloader, 
        model_forward_fn=model_forward_fn, 
        model_loss_fn=model_loss_fn, 
        metricsfn_dict=metricsfn_dict, 
        testing_tag='test',
        **kwargs
    )
    test_results = pd.DataFrame([validation_results]).to_csv(f"testing_{model.__class__.__name__}_{kwargs['ds_name']}.csv")
    return test_results

def evaluate_dataset(model, dataloader, model_forward_fn, model_loss_fn, device, metricsfn_dict, **kwargs):
    """
    Computes loss and metrics for a secondary dataset (Validation or Test).
    """
    model.eval()  # Set model to evaluation mode
    testing_tag = kwargs.get('testing_tag', 'val')
    temp_metricsfn_dict = {f"{testing_tag}_{k}":v for k,v in metricsfn_dict.items()}
    
    # Trackers for aggregation
    total_loss = 0.0
    metrics_results = {name: 0.0 for name in temp_metricsfn_dict.keys()}
    num_batches = 0
    
    # Disable gradient calculation for efficiency
    with torch.no_grad():
        for batch in dataloader:
            covariates, treatments, rewards, terminations, true_ites, mask = batch
            covariates = covariates.to(device)
            treatments = treatments.to(device)
            rewards = rewards.to(device)
            true_ites = true_ites.to(device)

            predictions = model_forward_fn(
                model=model, 
                treatments=treatments,
                covariates=covariates, 
                **kwargs
            )

            loss = model_loss_fn(predictions, rewards, treatments, **kwargs)
            total_loss += loss.item()
            
            for name, m_fn in temp_metricsfn_dict.items():
                metrics_results[name] += m_fn(model, treatments=treatments, covariates=covariates, true_effects=true_ites, mask=mask, **kwargs).item()
    
            num_batches += 1

    if num_batches == 0:
        return 0, 0, 0, 0
        
    avg_loss = total_loss / num_batches
    avg_metrics = {name: total / num_batches for name, total in metrics_results.items()}
    avg_metrics.update({f'{testing_tag}_avg_loss': avg_loss})

    return avg_metrics




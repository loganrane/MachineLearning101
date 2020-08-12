# create engine.py
import torch
import torch.nn as nn

def train(data_loader, model, optimizer, device):
    """
    This is the main training function that trains model for one epoch
    :param data_loader: this is the torch dataloader
    :param model: model (lstm model)
    :param optimizer: torch optimizer, eg adam, sgd etc.
    :param device: this can be 'cuda' or 'cpu'
    """
    
    # Set the model to training mode
    model.train()
    
    # Go through the batches of data in data loader
    for data in data_loader:
        # fetch review and target from the dict
        reviews = data['review']
        targets = data['target']
        
        # move the data to device that we want to use
        reviews = reviews.to(device, dtype = torch.long)
        targets = targets.to(device, dtype = torch.float)
        
        # clear the gradients 
        optimizer.zero_grad()
        
        # make predictions from the model
        predictions = model(reviews)
        
        # calculate the loss
        loss = nn.BCEWithLogitsLoss()(
            predictions, 
            targets.view(-1, 1)
        )
        
        # compute gradient of loss w.r.t.
        # all parameters of the model that are trainable
        loss.backward()
        
        # single optimization step
        optimizer.step()
        
        
    def evaluate(data_loader, model, device):
        # Initialize empty list to store predictions and targets
        final_predictions = []
        final_targets = []
        
        # Put the model in eval mode
        model.eval()
        
        # Disable gradient calculation
        with torch.no_grad():
            for data in data_loader:
                reviews = data['review']
                targets = data['target']
                reviews = reviews.to(device, dtype = torch.long)
                targets = targets.to(device, dtype = torch.float)
                
                # make predicitons
                predictions = model(reviews)
                
                # move predictions and targets to list
                predictions = predictions.cpu().numpy().tolist()
                targets = data['target'].cpu().numpy().tolist()
                final_predictions.extend(predictions)
                final_targets.extend(targets)
                
        # return final predictions and final targets
        return final_predictions, final_targets
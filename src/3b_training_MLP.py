import pandas as pd
import numpy as np
import joblib
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.init as init
from sklearn.metrics import mean_squared_error, r2_score

@torch.no_grad() # Add decorator to exclude the gradient in calculation
def get_inference(model, criterion, type):
  '''Get the model accuracy'''
  # Map the data loader
  data_loader = {
    'train': train_loader,
    'valid': valid_loader,
    'test': test_loader
  }[type]

  total_loss = 0

  model.eval()
  for i, (inputs, targets) in enumerate(data_loader):
    # Predict
    outputs = model(inputs)

    # Calcualte the loss
    total_loss += criterion(outputs, targets.unsqueeze(1)).item()

  # Find the mean loss
  loss = total_loss/len(data_loader)

  return total_loss


def train(model, criterion, max_epoch, train_loader, optimizer):
  '''Train the model'''
  # Init
  train_losses_list = []
  valid_losses_list = []
  losses = []

  # Start Iteration
  start_time = time.time()
  for epoch in range(max_epoch):
    # Training -- iterate over batch
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
      # Forward pass
      outputs = model(inputs)
      loss = criterion(outputs, targets.unsqueeze(1))

      # Backward and optimize
      optimizer.zero_grad()
      loss.backward()

      # Update the model parameter
      optimizer.step()

      # Log the minibatch (inner iteration)
      losses.append(loss.item())
      if not batch_idx % int(len(train_loader)/2.):
        print(f'Epoch: {epoch+1:03d}/{max_epoch:03d} '
              f'| Batch: {batch_idx:04d}/{len(train_loader):04d} '
              f'| Loss: {loss:.4f}')

    # Inference and evaluate model
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        train_loss = get_inference(model, criterion, 'train')
        valid_loss = get_inference(model, criterion, 'valid')

        print(f'Epoch: {epoch+1:03d}/{max_epoch:03d} '
              f'| Train: {train_loss:.4f}'
              f'| Valid: {valid_loss:.4f}')
        print('')

        train_losses_list.append(train_loss)
        valid_losses_list.append(valid_loss)
  
  # Finalize
  elapsed_time = (time.time() - start_time)
  print(f'Total training time: {elapsed_time/60:.0f} min {elapsed_time%60:.2f} sec')

  return losses, train_losses_list, valid_losses_list

# Create a config class
@dataclass
class ModelConfig:
    # Config model
    n_in: int = 14                # input size
    n_hidden: int = 250           # hidden layer size
    n_out: int = 1                # output size (1 for regeression case)
    
    # Config train
    lr: float = 0.1             # learning rates
    max_epochs: int = 100        # maximum epochs

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Create the MLP
        torch.manual_seed(42)

        # Flatten layer
        self.flatten = nn.Flatten()

        # 1st hidden layer
        self.fc1 = nn.Linear(config.n_in, config.n_hidden)
        init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')

        # ReLU activation function
        self.relu = nn.ReLU()

        # Output layer
        self.fc2 = nn.Linear(config.n_hidden, config.n_out)
        init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, X):
        X = self.flatten(X)
        X = self.relu(self.fc1(X))
        pred_density = self.fc2(X)
        return pred_density

def create_MLP_model(train_loader, valid_loader, test_loader):
  CONFIG = ModelConfig()
  CONFIG.n_in = X_train.shape[1]
  # Initialize models
  model_mlp = MLP(CONFIG)

  # Print number of parameters
  print('Total parameters :', sum(p.nelement() for p in model_mlp.parameters()))

  # Define optimizer
  optimizer_adam = torch.optim.Adam(params = model_mlp.parameters(),
                                lr = CONFIG.lr,
                                betas = (0.9, 0.999))
  
  # Define Loss Function
  criterion = nn.MSELoss()

  losses_mlp, train_loss_mlp, valid_loss_mlp = train(model=model_mlp,
                                                  criterion=criterion,
                                                  max_epoch=CONFIG.max_epochs,
                                                  train_loader=train_loader,
                                                  optimizer=optimizer_adam)
  # print("Training result:")
  # print("Loss:", losses_mlp)
  # print("Training Loss:", train_loss_mlp)
  # print("valid Loss:", valid_loss_mlp)

  # Evaluation on test set
  model_mlp.eval()
  y_pred_list = []
  y_test = []

  with torch.no_grad():
    for inputs, targets in test_loader:
      y_test_pred = model_mlp(inputs)
      y_test.extend(targets.unsqueeze(0)[0].tolist())
      y_pred_list.append(y_test_pred.numpy())

  y_test = np.array(y_test)

  y_pred_mlp = np.vstack(y_pred_list).flatten()

  mse = mean_squared_error(y_test, y_pred_mlp)
  rmse = np.sqrt(mse)
  aard = 100*np.mean(np.abs(y_pred_mlp - y_test) / y_test)
  r2 = r2_score(y_test, y_pred_mlp)

  print("Mean Squared Error:", mse)
  print("Root Mean Squared Error:", rmse)
  print("AARD:", aard)
  print("R² Score:", r2)

  return model_mlp

if __name__=='__main__':
  X_train = joblib.load('pickles/X_train.pkl')
  train_loader = joblib.load('pickles/train_loader.pkl')
  valid_loader = joblib.load('pickles/valid_loader.pkl')
  test_loader = joblib.load('pickles/test_loader.pkl')

  linreg_model = create_MLP_model(train_loader, valid_loader, test_loader)
  joblib.dump(linreg_model, 'pickles/model_mlp.pkl')
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import pandas as pd

def create_data_loader(dataset_loc: str):
  df = pd.read_csv(dataset_loc)
  # Assuming df is your DataFrame
  # Binning Pressure and Temperature
  # This Pressure and Temperature label will be use to stratify our data split process
  pressure_bins = pd.cut(df['Pressure'], bins=5, labels=False)
  temperature_bins = pd.cut(df['Temperature'], bins=5, labels=False)
  stratify_labels = pressure_bins.astype(str) + '_' + temperature_bins.astype(str)

  # Since our data is already on memory (DataFrame) we will split it right away (not using Tensor Sampler)
  # Splitting the dataset into 60% train, 20% validation, 20% test
  train_df, temp_df = train_test_split(df, test_size=0.4, stratify=stratify_labels, random_state=42)
  valid_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=stratify_labels[temp_df.index], random_state=42)

  X_train = train_df.drop(['Density', 'NG_TYPE'], axis=1).values
  y_train = train_df['Density'].values
  X_valid = valid_df.drop(['Density', 'NG_TYPE'], axis=1).values
  y_valid = valid_df['Density'].values
  X_test = test_df.drop(['Density', 'NG_TYPE'], axis=1).values
  y_test = test_df['Density'].values

  joblib.dump(X_train, 'pickles/X_train.pkl')
  joblib.dump(y_train, 'pickles/y_train.pkl')
  joblib.dump(X_valid, 'pickles/X_valid.pkl')
  joblib.dump(y_valid, 'pickles/y_valid.pkl')
  joblib.dump(X_test, 'pickles/X_test.pkl')
  joblib.dump(y_test, 'pickles/y_test.pkl')

  # Calculate mean and standard deviation for Pressure and Temperature in the training data
  pressure_mean = X_train[:, 0].mean()
  pressure_std = X_train[:, 0].std()
  temperature_mean = X_train[:, 1].mean()
  temperature_std = X_train[:, 1].std()

  feature_stats = {
      'pressure_mean': pressure_mean,
      'pressure_std': pressure_std,
      'temperature_mean': temperature_mean,
      'temperature_std': temperature_std
  }

  joblib.dump(feature_stats, 'pickles/feature_stats.pkl')

  # Apply normalization to Pressure and Temperature in all sets (train, validation, and test)
  X_train_normalized = X_train.copy()
  X_train_normalized[:, 0] = (X_train[:, 0] - pressure_mean) / pressure_std
  X_train_normalized[:, 1] = (X_train[:, 1] - temperature_mean) / temperature_std

  X_valid_normalized = X_valid.copy()
  X_valid_normalized[:, 0] = (X_valid[:, 0] - pressure_mean) / pressure_std
  X_valid_normalized[:, 1] = (X_valid[:, 1] - temperature_mean) / temperature_std

  X_test_normalized = X_test.copy()
  X_test_normalized[:, 0] = (X_test[:, 0] - pressure_mean) / pressure_std
  X_test_normalized[:, 1] = (X_test[:, 1] - temperature_mean) / temperature_std

  # Convert the normalized data to PyTorch tensors
  X_train_tensor = torch.tensor(X_train_normalized, dtype=torch.float32)
  X_valid_tensor = torch.tensor(X_valid_normalized, dtype=torch.float32)
  X_test_tensor = torch.tensor(X_test_normalized, dtype=torch.float32)
  y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
  y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32)
  y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

  # Define params
  batch_size = 64
  torch.manual_seed(42)

  # Create TensorDatasets
  train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
  valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
  test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

  # Create Tensor DataLoaders
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

  return train_loader, valid_loader, test_loader

if __name__=='__main__':
  dataset_loc = 'dataset/ng_density_all.csv'
  train_loader, valid_loader, test_loader = create_data_loader(dataset_loc)
  joblib.dump(train_loader, 'pickles/train_loader.pkl')
  joblib.dump(valid_loader, 'pickles/valid_loader.pkl')
  joblib.dump(test_loader, 'pickles/test_loader.pkl')
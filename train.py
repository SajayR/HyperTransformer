import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from hyper_transformer_main import HyperTransformer
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import glob
import tqdm
import warnings
import wandb
from torch.autograd import gradcheck, detect_anomaly

warnings.simplefilter(action='ignore', category=FutureWarning)

# Hyperparameters
input_size = 1
hidden_size = 256
num_layers = 2
output_size = 1
patch_size = 32
d_model = 256
num_heads = 8
d_ff = 2048
dropout = 0.1
max_seq_length = 5000
lstm_input_len = 128
pred_len = 24
batch_size = 32
num_epochs = 100
learning_rate = 0.0001
clip_value = 0.4

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

class PM25Dataset(Dataset):
    def __init__(self, file_paths, seq_len, pred_len, column_name='PM2.5 (µg/m³)'):
        self.file_paths = file_paths
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.column_name = column_name
        self.scaler = StandardScaler()
        self.data = self.load_and_preprocess_data()

    def load_and_preprocess_data(self):
        all_data = []
        for file_path in self.file_paths:
            df = pd.read_csv(file_path)
            if self.column_name in df.columns:
                data = pd.to_numeric(df[self.column_name], errors='coerce')
                data = data.interpolate(method='linear', limit_direction='both')
                data = data.fillna(method='ffill')
                data = data.fillna(method='bfill')
                if not data.isna().any():
                    all_data.extend(data.values)
        
        if not all_data:
            raise ValueError("No valid data found in the provided CSV files.")
        
        all_data = np.array(all_data).reshape(-1, 1)
        self.scaler.fit(all_data)
        return self.scaler.transform(all_data).flatten()

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+self.seq_len:idx+self.seq_len+self.pred_len]
        return torch.FloatTensor(x), torch.FloatTensor(y)

def prepare_data(data_folder, seq_len, pred_len, test_size=0.2):
    file_list = glob.glob(os.path.join(data_folder, '*.csv'))
    train_files, test_files = train_test_split(file_list, test_size=test_size, random_state=42)

    train_dataset = PM25Dataset(train_files, seq_len, pred_len)
    test_dataset = PM25Dataset(test_files, seq_len, pred_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset.scaler

def train_hypertransformer(model, train_loader, val_loader, num_epochs, learning_rate, device, scaler):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            x = x.squeeze(-1)
            
            optimizer.zero_grad()
            output = model(x)
            y = y.unsqueeze(-1)
            loss = criterion(output, y)

            y_unscaled = scaler.inverse_transform(y.cpu().numpy().reshape(-1, 1)).reshape(y.shape)
            output_unscaled = scaler.inverse_transform(output.cpu().detach().numpy().reshape(-1, 1)).reshape(output.shape)
            scaled_loss = np.mean((output_unscaled - y_unscaled) ** 2)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
            optimizer.step()

            total_loss += loss.item()

            # Log batch-level metrics
            wandb.log({
                "batch_loss": loss.item(),
                "batch_scaled_loss": scaled_loss
            })

        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        val_mse, val_mae = validate(model, val_loader, criterion, device, scaler)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val MSE: {val_mse:.4f}, Val MAE: {val_mae:.4f}")
        
        # Log epoch-level metrics
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_mse": val_mse,
            "val_mae": val_mae,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        # Learning rate scheduling
        scheduler.step(val_mse)

    print("Training completed!")

def validate(model, val_loader, criterion, device, scaler):
    model.eval()
    total_mse = 0
    total_mae = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm.tqdm(val_loader, desc="Validating"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            x = x.squeeze(-1)
            output = model(x)
            
            mse_loss = criterion(output, y.unsqueeze(-1))
            total_mse += mse_loss.item() * y.size(0)
            
            output_unscaled = scaler.inverse_transform(output.cpu().numpy().reshape(-1, 1)).reshape(output.shape)
            y_unscaled = scaler.inverse_transform(y.cpu().numpy().reshape(-1, 1)).reshape(y.shape[0], -1)
            
            output_unscaled = output_unscaled.squeeze(-1)
            mae_loss = np.mean(np.abs(output_unscaled - y_unscaled))
            total_mae += mae_loss * y.size(0)
            
            total_samples += y.size(0)
    
    avg_mse = total_mse / total_samples
    avg_mae = total_mae / total_samples
    
    return avg_mse, avg_mae

def check_gradients(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: grad exists: {param.grad is not None}, grad norm: {param.grad.norm().item() if param.grad is not None else 'N/A'}")

if __name__ == "__main__":
    # Initialize wandb
    wandb.init(project="hypertransformer", config={
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "output_size": output_size,
        "patch_size": patch_size,
        "d_model": d_model,
        "num_heads": num_heads,
        "d_ff": d_ff,
        "dropout": dropout,
        "max_seq_length": max_seq_length,
        "lstm_input_len": lstm_input_len,
        "pred_len": pred_len,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "clip_value": clip_value
    })

    # Create model
    model = HyperTransformer(
        input_size, hidden_size, num_layers, output_size,
        patch_size, d_model, num_heads, d_ff, dropout, max_seq_length, lstm_input_len, pred_len
    )
    print("Model created successfully.")
    print("Model layers:")
    for name, module in model.named_children():
        print(f"{name}: {module}")

    # Prepare data
    data_folder = "/home/srip24_llm/.ht/cleaneddatasets"  # Replace with the actual path to your CSV files
    train_loader, test_loader, scaler = prepare_data(data_folder, lstm_input_len, pred_len)
    print("Data prepared successfully.")

    train_hypertransformer(model, train_loader, test_loader, num_epochs, learning_rate, device, scaler)

    # Test the model
    model.eval()
    test_mse, test_mae = validate(model, test_loader, nn.MSELoss(), device, scaler)
    print(f"Test MSE: {test_mse:.4f}, Test MAE: {test_mae:.4f}")
    
    # Log final test metrics
    wandb.log({
        "test_mse": test_mse,
        "test_mae": test_mae
    })

    # Verify that LSTM weights are generated anew each time
    x_test = next(iter(test_loader))[0][:1].to(device)
    output1 = model(x_test)
    output2 = model(x_test)
    outputs_different = not torch.allclose(output1, output2)
    print(f"Outputs are different (weights generated anew): {outputs_different}")

    # Log the verification result
    wandb.log({
        "outputs_different": outputs_different
    })

    # Save the model
    torch.save(model.state_dict(), "hypertransformer_model.pth")
    print("Model saved successfully.")

    # Finish the wandb run
    wandb.finish()
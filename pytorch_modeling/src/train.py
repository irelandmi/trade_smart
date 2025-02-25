# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import mlflow
import mlflow.pytorch  # For PyTorch-specific logging
mlflow.set_tracking_uri("http://127.0.0.1:5000")
from model import MyNet
from utils.dataloader import MyDataset, load_and_preprocess_data

def train_model(csv_path):
    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler = load_and_preprocess_data(csv_path)

    # Create datasets/loaders
    train_dataset = MyDataset(X_train, y_train)
    val_dataset = MyDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Hyperparams
    input_dim = X_train.shape[1]
    hidden_dim = 64
    output_dim = 1
    lr = 1e-3
    epochs = 5

    # Start an MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("input_dim", input_dim)
        mlflow.log_param("hidden_dim", hidden_dim)
        mlflow.log_param("output_dim", output_dim)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("epochs", epochs)

        # Initialize model, criterion, optimizer
        model = MyNet(input_dim, hidden_dim, output_dim)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Training loop
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.float()
                y_batch = y_batch.float().view(-1, 1)

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.float()
                    y_batch = y_batch.float().view(-1, 1)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()

            # Average losses
            avg_train_loss = running_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            # Log metrics to MLflow
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

        # Evaluate on test set at the end
        test_loss = evaluate_on_test(model, (X_test, y_test))
        mlflow.log_metric("test_loss", test_loss)

        # (Optional) Log the scaler artifact
        import pickle
        with open("scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        mlflow.log_artifact("scaler.pkl", artifact_path="preprocessing")

        # Log the model to MLflow (PyTorch flavor)
        mlflow.pytorch.log_model(model, "model")
        print("Model training complete. Logged to MLflow!")

def evaluate_on_test(model, test_data):
    X_test, y_test = test_data
    X_test_t = torch.tensor(X_test).float()
    y_test_t = torch.tensor(y_test).float().view(-1, 1)
    criterion = nn.BCEWithLogitsLoss()

    model.eval()
    with torch.no_grad():
        outputs = model(X_test_t)
        loss = criterion(outputs, y_test_t)

    return loss.item()

if __name__ == "__main__":
    csv_path = "../data/raw/my_dataset.csv"
    train_model(csv_path)

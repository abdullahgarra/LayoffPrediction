import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Paths to datasets
technical_data_path = "finalDataset.csv"
embedding_data_path = "finalDataset_with_embeddings.csv"

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define PyTorch model for technical data + precomputed embeddings
class CombinedModel(nn.Module):
    def __init__(self, technical_input_dim, embedding_dim=768):
        super(CombinedModel, self).__init__()
        # LSTM for technical data
        self.lstm = nn.LSTM(input_size=technical_input_dim, hidden_size=512, num_layers=2, batch_first=True, bidirectional=True)
        self.fc_technical = nn.Sequential(
            nn.Linear(512 * 2, 128),  # Bidirectional
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2)
        )

        # Dense layers for combined features
        self.fc_combined = nn.Sequential(
            nn.Linear(128 + embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)  # Regression output
        )

    def forward(self, technical_data, embeddings):
        # Process technical data through LSTM
        lstm_out, _ = self.lstm(technical_data)
        lstm_out = lstm_out[:, -1, :]  # Take the last hidden state
        technical_features = self.fc_technical(lstm_out)

        # Concatenate with embeddings
        combined_features = torch.cat((technical_features, embeddings), dim=1)
        return self.fc_combined(combined_features)

# Training loop
def train_model(model, dataloader, criterion, optimizer, num_epochs=15):
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in dataloader:
            technical_data, embeddings, labels = [x.to(device) for x in batch]

            optimizer.zero_grad()
            outputs = model(technical_data, embeddings)
            loss = criterion(outputs.view(-1), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * technical_data.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Main function
def main():
    # Load technical and embedding datasets
    technical_data = pd.read_csv(technical_data_path, index_col=0)
    embedding_data = pd.read_csv(embedding_data_path)

    # Merge datasets on the layoff date
    merged_data = pd.merge(technical_data, embedding_data, left_on="Date_layoffs", right_on="Date_layoffs", how="inner")

    # Handle potential NaN values in the average_embedding column
    embedding_dim = 768  # Set the embedding dimension
    merged_data["average_embedding"] = merged_data["average_embedding"].fillna(str([0.0] * embedding_dim))

    # Safely evaluate strings into lists
    def safe_eval(x):
        try:
            return eval(x)
        except:
            return [0.0] * embedding_dim  # Default embedding

    merged_data["average_embedding"] = merged_data["average_embedding"].apply(safe_eval)

    # Ensure all embeddings are of correct dimension
    def ensure_correct_dimension(embedding, expected_dim=768):
        if len(embedding) != expected_dim:
            print(f"Incorrect embedding dimension detected. Padding/truncating to {expected_dim}.")
            return embedding[:expected_dim] + [0.0] * (expected_dim - len(embedding))
        return embedding

    merged_data["average_embedding"] = merged_data["average_embedding"].apply(lambda x: ensure_correct_dimension(x, embedding_dim))

    # Convert embeddings to stacked arrays
    X_embeddings = np.stack(merged_data["average_embedding"].values)
    print(f"Embedding data shape: {X_embeddings.shape}")
    assert X_embeddings.shape[1] == embedding_dim, "Embeddings dimension mismatch!"

    # Prepare technical data and labels
    X_technical = merged_data.drop(columns=["Percentage", "average_embedding", "Date_layoffs"]).values
    y = merged_data["Percentage"].values

    # Split data chronologically into training and test sets
    split_index = int(0.8 * len(X_technical))
    X_technical_train, X_technical_test = X_technical[:split_index], X_technical[split_index:]
    X_embeddings_train, X_embeddings_test = X_embeddings[:split_index], X_embeddings[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Convert data to tensors
    X_technical_train = torch.tensor(X_technical_train, dtype=torch.float32)
    X_technical_test = torch.tensor(X_technical_test, dtype=torch.float32)
    X_embeddings_train = torch.tensor(X_embeddings_train, dtype=torch.float32)
    X_embeddings_test = torch.tensor(X_embeddings_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Create PyTorch datasets and dataloaders
    train_dataset = torch.utils.data.TensorDataset(X_technical_train.unsqueeze(1), X_embeddings_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_technical_test.unsqueeze(1), X_embeddings_test, y_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Initialize model, criterion, optimizer
    model = CombinedModel(technical_input_dim=X_technical.shape[1]).to(device)
    criterion = nn.L1Loss()  # MAE
    optimizer = optim.Adam(model.parameters(), lr=5e-3)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs=1000)

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        predictions = []
        actuals = []
        for batch in test_loader:
            technical_data, embeddings, labels = [x.to(device) for x in batch]
            outputs = model(technical_data, embeddings)
            predictions.extend(outputs.view(-1).cpu().numpy())
            actuals.extend(labels.cpu().numpy())
    
    # Calculate MAE
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    mae = np.mean(np.abs(predictions - actuals))
    print(f"Test MAE: {mae:.4f}")

if __name__ == "__main__":
    main()

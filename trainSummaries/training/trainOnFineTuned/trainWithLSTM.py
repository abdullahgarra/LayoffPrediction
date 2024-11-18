import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np

# Paths to datasets
technical_data_path = "finalDataset.csv"
textual_data_path = "filtered_articles_with_percentages.csv"

# Paths to fine-tuned FinBERT models for each window
finetuned_models = {
    "filtered_7_days": "finbert_output_7_days",
    "filtered_15_days": "finbert_output_15_days",
    "filtered_30_days": "finbert_output_30_days",
    "filtered_90_days": "finbert_output_90_days"
}

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load fine-tuned FinBERT model
def load_finetuned_finbert_model(model_path):
    return AutoModel.from_pretrained(model_path)

# Tokenizer for FinBERT
model_name = "ProsusAI/finbert"
finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)

# Get FinBERT embedding for text
def get_finbert_embedding(text, finbert_model):
    inputs = finbert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = finbert_model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
    return cls_embedding.cpu().numpy()

# Define PyTorch model for technical data + FinBERT
class CombinedModel(nn.Module):
    def __init__(self, technical_input_dim, finbert_embedding_dim=768):
        super(CombinedModel, self).__init__()
        # LSTM for technical data
        self.lstm = nn.LSTM(input_size=technical_input_dim, hidden_size=512, num_layers=2, batch_first=True, bidirectional=True)
        self.fc_technical = nn.Sequential(
            nn.Linear(512*2, 128),  # Bidirectional
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2)
        )

        # Dense layers for combined features
        self.fc_combined = nn.Sequential(
            nn.Linear(128 + finbert_embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)  
        )

    def forward(self, technical_data, finbert_embeddings):
        # Process technical data through LSTM
        lstm_out, _ = self.lstm(technical_data)
        lstm_out = lstm_out[:, -1, :]  # Take the last hidden state
        technical_features = self.fc_technical(lstm_out)

        # Concatenate with FinBERT embeddings
        combined_features = torch.cat((technical_features, finbert_embeddings), dim=1)
        return self.fc_combined(combined_features)

# Training loop
def train_model(model, dataloader, criterion, optimizer, num_epochs=50):
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in dataloader:
            technical_data, finbert_embeddings, labels = [x.to(device) for x in batch]

            optimizer.zero_grad()
            outputs = model(technical_data, finbert_embeddings)
            loss = criterion(outputs.view(-1), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * technical_data.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Process and train on each window
def process_window(window_name, technical_data, textual_data):
    # Load fine-tuned FinBERT for the current window
    finbert_model_path = finetuned_models[window_name]
    print(f"Loading fine-tuned FinBERT model for {window_name} from {finbert_model_path}...")
    finbert_model = load_finetuned_finbert_model(finbert_model_path).to(device)

    # Filter the textual data by window
    window_data = textual_data[textual_data['table_name'] == window_name]

    # Merge technical data and textual data on `layoff_id`
    merged_data = pd.merge(technical_data, window_data, left_index=True, right_on="layoff_id")

    if 'Percentage_x' in merged_data.columns and 'Percentage_y' in merged_data.columns:
        merged_data = merged_data.rename(columns={'Percentage_x': 'Percentage'}).drop(columns=['Percentage_y'])
    
    X_technical = merged_data.drop(columns=["Percentage", "filtered_articles", "table_name"]).values
    y = merged_data["Percentage"].values
    X_text = merged_data["filtered_articles"]

    # Chronological split into train and test sets
    split_index = int(len(X_technical) * 0.8)  # 80% train, 20% test
    X_technical_train, X_technical_test = X_technical[:split_index], X_technical[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    X_text_train, X_text_test = X_text[:split_index], X_text[split_index:]

    # Convert data to tensors
    X_technical_train = torch.tensor(X_technical_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    finbert_embeddings_train = torch.tensor(np.vstack([get_finbert_embedding(text, finbert_model) for text in X_text_train]), dtype=torch.float32)

    train_dataset = torch.utils.data.TensorDataset(X_technical_train.unsqueeze(1), finbert_embeddings_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)

    # Prepare test data
    X_technical_test = torch.tensor(X_technical_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    finbert_embeddings_test = torch.tensor(np.vstack([get_finbert_embedding(text, finbert_model) for text in X_text_test]), dtype=torch.float32)

    # Initialize model, criterion, optimizer
    model = CombinedModel(technical_input_dim=X_technical.shape[1]).to(device)
    criterion = nn.L1Loss()  # MAE
    optimizer = optim.Adam(model.parameters(), lr=5e-3)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs=500)

    # Test the model
    model.eval()
    with torch.no_grad():
        outputs = model(X_technical_test.unsqueeze(1).to(device), finbert_embeddings_test.to(device))
        mae = mean_absolute_error(y_test.numpy(), outputs.cpu().numpy())
    print(f"Test MAE for {window_name}: {mae:.4f}")

    # Save the model
    model_save_path = f"./pytorch_model_{window_name}.pt"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model for {window_name} saved at {model_save_path}")

# Main function to process each window
def main():
    # Load technical and textual datasets
    technical_data = pd.read_csv(technical_data_path, index_col=0)
    textual_data = pd.read_csv(textual_data_path)

    # Process each window
    for window_name in ["filtered_7_days", "filtered_15_days", "filtered_30_days", "filtered_90_days"]:
        print(f"\nProcessing window: {window_name}")
        process_window(window_name, technical_data, textual_data)

if __name__ == "__main__":
    main()

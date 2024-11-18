import torch
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
from transformers import get_polynomial_decay_schedule_with_warmup
from datasets import Dataset
import pandas as pd
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import mean_absolute_error
import os
from sklearn.model_selection import train_test_split

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def save_model_and_tokenizer(model, tokenizer, path):
    # Save the PyTorch model
    torch.save(model.state_dict(), f"{path}/pytorch_model.bin")
    # Save the tokenizer
    tokenizer.save_pretrained(path)
    print(f"Model and tokenizer saved to {path}")


# Define custom FinBERT regression model
class FinBERTRegression(torch.nn.Module):
    def __init__(self, model_name):
        super(FinBERTRegression, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.regression_head = torch.nn.Linear(self.bert.config.hidden_size, 1)  # Output a single value for regression

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_output = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token for regression
        logits = self.regression_head(cls_output)
        
        loss = None
        if labels is not None:
            loss_fn = torch.nn.L1Loss()  
            loss = loss_fn(logits.view(-1), labels.view(-1)) 
           
        
        return SequenceClassifierOutput(loss=loss, logits=logits)

# Load FinBERT model and tokenizer
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize function for summaries
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

# Define MAE computation for regression
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.flatten()
    mae = mean_absolute_error(labels, predictions)
    return {"mae": mae}


def lr_scheduler(optimizer, num_training_steps):
    return get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
        power=3.0  
    )
# Main function to fine-tune model on each data window
def fine_tune_on_window(train_dataset, test_dataset, window_name, model):
    print(f"\nStarting fine-tuning for the {window_name} data window")

    # Tokenize the datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    # Remove only unused columns and set format for PyTorch
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # Set up training arguments with an initial learning rate
    training_args = TrainingArguments(
        output_dir=f"./finbert_output_{window_name}",
        eval_strategy="epoch",
        logging_strategy="steps",
        save_strategy="no",
        num_train_epochs=100,
        per_device_train_batch_size=8,
        logging_dir="./logs",
        logging_steps=100,
        learning_rate=5e-4
    )

    # Create a custom optimizer with learning rate scheduling
    num_training_steps = len(train_dataset) * training_args.num_train_epochs // training_args.per_device_train_batch_size
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)
    scheduler = lr_scheduler(optimizer, num_training_steps)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler)  
    )

    # Fine-tune the model
    trainer.train()

    # In your training code, replace model.save_pretrained with the custom save function
    model_save_path = f"./finbert_output_{window_name}"
    save_model_and_tokenizer(model, tokenizer, model_save_path)
    print(f"Model saved for {window_name} at {model_save_path}")

    # Evaluate the model on the test set
    test_results = trainer.evaluate(test_dataset)
    print(f"Final MAE on test set for {window_name} data window: {test_results['eval_mae']}")

def main():
    # Load the merged dataset
    merged_df = pd.read_csv("filtered_articles_with_percentages.csv")

    # Initialize the base FinBERT model once, and clone it for each training session
    model = FinBERTRegression(model_name)

    # Define different windows and corresponding datasets
    # windows = {
    #     "7_days": merged_df[merged_df['table_name'] == "filtered_7_days"],
    #     "15_days": merged_df[merged_df['table_name'] == "filtered_15_days"],
    #     "30_days": merged_df[merged_df['table_name'] == "filtered_30_days"],
    #     "90_days": merged_df[merged_df['table_name'] == "filtered_90_days"]
    # }

    windows = {"90_days": merged_df[merged_df['table_name'] == "filtered_90_days"]}
    # Fine-tune FinBERT on each time window dataset
    for window_name, data in windows.items():
        print(f"Processing {window_name} window with {len(data)} records.")
        
        # Split data into train and test sets (80% train, 20% test)
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
        train_dataset = Dataset.from_pandas(train_data[['filtered_articles', 'Percentage']].rename(columns={"filtered_articles": "text", "Percentage": "label"}))
        test_dataset = Dataset.from_pandas(test_data[['filtered_articles', 'Percentage']].rename(columns={"filtered_articles": "text", "Percentage": "label"}))
        
        # Fine-tune the model on each dataset
        fine_tune_on_window(train_dataset, test_dataset, window_name, FinBERTRegression(model_name))

if __name__ == "__main__": 
    main()

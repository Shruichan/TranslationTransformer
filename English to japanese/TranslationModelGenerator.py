import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import re
from transformers import EncoderDecoderModel, BertTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu
import matplotlib.pyplot as plt

# Download nltk resources
nltk.download('punkt')
# Read dataset from file
file_path = "jpn.txt"  

def clean_sentence(sentence):
    # Split the sentence by English or Japanese period, or question mark
    parts = re.split(r'\.|\?|\|\！|\？|\!', sentence)
    # Keep only the part before the Japanese period
    cleaned_sentence = parts[0].strip()
    return cleaned_sentence


data = {"English": [], "Japanese": []}

with open(file_path, 'r', encoding='utf-8') as file:
    for line in file.readlines():
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            english_sentence = clean_sentence(parts[0])
            japanese_sentence = clean_sentence(parts[1])
            data["English"].append(english_sentence)
            data["Japanese"].append(japanese_sentence)

df = pd.DataFrame(data)

class TranslationDataset(Dataset):
    def __init__(self, source_sentences, target_sentences, tokenizer_source, tokenizer_target, max_length=128):
        self.tokenizer_source = tokenizer_source
        self.tokenizer_target = tokenizer_target
        self.max_length = max_length

        # Tokenizing data in initialization for better performance
        self.source_encodings = tokenizer_source(source_sentences, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt")
        self.target_encodings = tokenizer_target(target_sentences, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt")

    def __len__(self):
        return len(self.source_encodings['input_ids'])

    def __getitem__(self, idx):
        return {
            'input_ids': self.source_encodings['input_ids'][idx],
            'attention_mask': self.source_encodings['attention_mask'][idx],
            'labels': self.target_encodings['input_ids'][idx],
            'decoder_input_ids': self.target_encodings['input_ids'][idx]
        }




# Set device to GPU if available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load pre-trained tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")


# Split data into train and validation sets
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

train_dataset = TranslationDataset(train_df["English"].tolist(), train_df["Japanese"].tolist(), tokenizer, tokenizer)
val_dataset = TranslationDataset(val_df["English"].tolist(), val_df["Japanese"].tolist(), tokenizer, tokenizer)

# Move model to GPU
# Load pre-trained models for both encoder and decoder
model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-multilingual-cased", "bert-base-multilingual-cased")

# Set the decoder_start_token_id
model.config.decoder_start_token_id = tokenizer.cls_token_id  # CLS token is often used as a start token for BERT-based models

# Set other necessary configurations, e.g., pad_token_id if not already set
model.config.pad_token_id = tokenizer.pad_token_id

model.to(device)



# Set batch size, epochs, and optimizer
batch_size = 16
epochs = 3
learning_rate = 5e-5
optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
# Create DataLoader for training and validation
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
# Check DataLoader
print("Length of train DataLoader:", len(train_loader))
print("Length of val DataLoader:", len(val_loader))

# Iterate over a few batches manually to check data loading
for batch_idx, batch in enumerate(train_loader):
    if batch_idx >= 3:
        break
    print("Batch", batch_idx, "Input IDs shape:", batch['input_ids'].shape)
    print("Batch", batch_idx, "Attention Mask shape:", batch['attention_mask'].shape)
    print("Batch", batch_idx, "Decoder Input IDs shape:", batch['decoder_input_ids'].shape)

# Check Tokenization
for idx in range(3):
    sample = train_dataset[idx]
    print("Sample", idx, "Input IDs shape:", sample['input_ids'].shape)
    print("Sample", idx, "Attention Mask shape:", sample['attention_mask'].shape)
    print("Sample", idx, "Decoder Input IDs shape:", sample['decoder_input_ids'].shape)
    print("Sample", idx, "Input IDs:", sample['input_ids'])
    print("Sample", idx, "Decoder Input IDs:", sample['decoder_input_ids'])

# Check Model and Optimizer
print("Model:", model)
print("Optimizer:", optimizer)


# Prepare to save training and validation results
results_path = "training_results.txt"
with open(results_path, "w") as f:
    f.write("Training and Validation Results\n")

# Initialize lists to store losses for plotting
training_losses = []
validation_losses = []

# Training loop
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}...")
    model.train()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        # Calculate accuracy
        preds = torch.argmax(outputs.logits, dim=-1)
        not_ignore = labels != tokenizer.pad_token_id
        correct = (preds == labels) & not_ignore
        total_correct += correct.sum().item()
        total_tokens += not_ignore.sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate batch accuracy
        batch_accuracy = correct.sum().item() / not_ignore.sum().item()

        print(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}/{len(train_loader)}, Training Loss: {loss.item()}, Accuracy: {batch_accuracy:.4f}")
        with open(results_path, "a") as f:
            f.write(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}/{len(train_loader)}, Training Loss: {loss.item()}, Accuracy: {batch_accuracy:.4f}\n")

    avg_train_loss = total_loss / len(train_loader)
    avg_train_accuracy = total_correct / total_tokens
    training_losses.append(avg_train_loss)
    print(f'Epoch {epoch + 1}/{epochs}, Average Train Loss: {avg_train_loss}, Average Accuracy: {avg_train_accuracy:.4f}')
    with open(results_path, "a") as f:
        f.write(f'Epoch {epoch + 1}/{epochs}, Average Train Loss: {avg_train_loss}, Average Accuracy: {avg_train_accuracy:.4f}\n')

    # Validation loop
    model.eval()
    val_loss = 0
    val_correct = 0
    val_tokens = 0
    references = []
    hypotheses = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()

            # Decoding the labels (ground truth) and predictions
            preds = torch.argmax(outputs.logits, dim=-1)
            not_ignore = labels != tokenizer.pad_token_id
            correct = (preds == labels) & not_ignore
            val_correct += correct.sum().item()
            val_tokens += not_ignore.sum().item()

            # Calculate batch accuracy
            batch_accuracy = correct.sum().item() / not_ignore.sum().item()

            print(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}/{len(val_loader)}, Validation Loss: {loss.item()}, Accuracy: {batch_accuracy:.4f}")
            with open(results_path, "a") as f:
                f.write(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}/{len(val_loader)}, Validation Loss: {loss.item()}, Accuracy: {batch_accuracy:.4f}\n")

    avg_val_loss = val_loss / len(val_loader)
    avg_val_accuracy = val_correct / val_tokens
    validation_losses.append(avg_val_loss)
    print(f'Epoch {epoch + 1}/{epochs}, Average Validation Loss: {avg_val_loss}, Average Accuracy: {avg_val_accuracy:.4f}')
    with open(results_path, "a") as f:
        f.write(f'Epoch {epoch + 1}/{epochs}, Average Validation Loss: {avg_val_loss}, Average Accuracy: {avg_val_accuracy:.4f}\n')





model_save_path = "bert_translation_model_two.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
with open(results_path, "a") as f:
    f.write(f"Model saved to {model_save_path}\n")

print("Training finished.")


# After validation loop
bleu_score = corpus_bleu(references, hypotheses)
print(f'BLEU Score: {bleu_score}')

# Save the BLEU score along with other results
with open(results_path, "a") as f:
    f.write(f'Validation Loss: {avg_val_loss}\n')
    f.write(f'BLEU Score: {bleu_score}\n')

# Plotting the loss progression
plt.figure(figsize=(10, 5))
plt.plot(training_losses, label='Training Loss')
plt.plot(validation_losses, label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_progression.png')
plt.show()


# Save the trained model


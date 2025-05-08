#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DistilBERT model for multi-label job skill classification from job summaries.
"""

import csv
import torch
import numpy as np
import joblib
import re
import os
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
from torch.optim import AdamW
import matplotlib.pyplot as plt  # REQUIRED FOR PLOTTING
import random


# Compute token-based Jaccard similarity between true and predicted labels
# y_true: array of true label vectors (binary multi-hot format)
# y_pred: array of predicted label vectors (binary multi-hot format)
# mlb_classes: list of skill names corresponding to label indices
def jaccard_similarity(y_true, y_pred, mlb_classes):
    scores = []
    for true_row, pred_row in zip(y_true, y_pred):
        true_skills = [mlb_classes[i] for i in np.where(true_row == 1)[0]]
        pred_skills = [mlb_classes[i] for i in np.where(pred_row == 1)[0]]
        overlap = []
        for t in true_skills:
            for p in pred_skills:
                tokens_t = set(re.findall(r'\w+', t.lower()))  # Tokenize and lowercase
                tokens_p = set(re.findall(r'\w+', p.lower()))  # Tokenize and lowercase
                if tokens_t or tokens_p:
                    overlap.append(len(tokens_t & tokens_p) / len(tokens_t | tokens_p))  # Jaccard-like overlap
        if overlap:
            scores.append(np.mean(overlap))  # Average overlap per true/pred skill pair
    return np.mean(scores) if scores else 0.0  # Return the mean overlap score


# Custom PyTorch dataset for job summaries and multi-label skills
# summaries: list of job summaries (text)
# labels: binary label matrix (multi-label binarized skills)
# tokenizer: a HuggingFace tokenizer instance
# max_len: maximum sequence length for tokenization
class JobSkillDataset(Dataset):
    def __init__(self, summaries, labels, tokenizer, max_len=256):
        self.summaries = summaries
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.summaries)

    def __getitem__(self, idx):
        summary = str(self.summaries[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            summary,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

# Load job summary and skill data from a CSV file
# file_path: path to the CSV file with 'job_summary' and 'filtered_skills' columns
def load_data_from_csv(file_path):
    summaries = []
    skill_lists = []
    with open(file_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'filtered_skills' in row and 'job_summary' in row:
                summaries.append(row['job_summary'])
                skill_lists.append(row['filtered_skills'].split(','))
    return summaries, skill_lists

# Train a DistilBERT-based model on job summary data
# dataset: PyTorch Dataset containing tokenized job summaries and labels
# model: HuggingFace DistilBERT model with classification head
# tokenizer: tokenizer used for encoding summaries
# mlb: MultiLabelBinarizer instance used for transforming labels
# save_dir: path to directory where model and plots will be saved
# epochs: number of training epochs (default = 3)
# batch_size: size of batches for training and evaluation (default = 8)
# learning_rate: learning rate for optimizer (default = 5e-5)
# threshold: threshold to binarize predicted probabilities (default = 0.3)
def train_model(dataset, model, tokenizer, mlb, save_dir, epochs=3, batch_size=8, learning_rate=5e-5, threshold=0.3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # split dataset into train, test and validate datasets
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    cpu_count = os.cpu_count() #use max num of CPU cores to load data
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=cpu_count-1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=cpu_count-1)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_jaccards = []  

    os.makedirs(save_dir, exist_ok=True)
    log_file_path = os.path.join(save_dir, "validation_predictions.log")
    log_file = open(log_file_path, "w", encoding="utf-8")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        print(f"\nEpoch {epoch+1}/{epochs}")
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Average Training Loss: {avg_train_loss:.4f}")

        # evaluate on validation set
        model.eval()
        val_loss = 0
        correct_preds = 0
        total_preds = 0
        total_jaccard = 0 
        all_preds = []
        all_true = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()

                probs = torch.sigmoid(outputs.logits)
                preds = (probs > threshold).float()  # Binarize predictions (e.g THRESHOLD = 0.3)

                # Accumulate for metric computation
                all_preds.append(preds.cpu())
                all_true.append(labels.cpu())
                # Calculate the exact match for this batch
                correct_preds += torch.sum((preds == labels).all(dim=1)).item()
                total_preds += labels.size(0)

                # Calculate Jaccard similarity for batch,weight jaccard by batch size
                batch_jaccard = jaccard_similarity(labels.cpu().numpy(), preds.cpu().numpy(), mlb.classes_)
                total_jaccard += batch_jaccard * labels.size(0)  

                # Log a few examples to the file
                for i in range(min(3, len(labels))):  # Log only first 3 examples per batch to prevent overwhelming logs
                    expected_indices = labels[i].cpu().numpy().astype(int).nonzero()[0]
                    predicted_indices = preds[i].cpu().numpy().astype(int).nonzero()[0]

                    expected_skills = [mlb.classes_[idx] for idx in expected_indices]
                    predicted_skills = [mlb.classes_[idx] for idx in predicted_indices]

                    print(f"Expected Skills: {expected_skills}\n")
                    print(f"Predicted Skills: {predicted_skills}\n")
                    print(f"Jaccard Similarity: {batch_jaccard:.4f}\n")

                    print("-" * 80 + "\n")

        # Calculate and store validation metrics
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct_preds / total_preds
        avg_jaccard = total_jaccard / total_preds  

        # Store metrics for plotting
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        val_jaccards.append(avg_jaccard)


         # Compute metrics
        all_preds_tensor = torch.cat(all_preds).numpy()
        all_true_tensor = torch.cat(all_true).numpy()

        val_precision_micro = precision_score(all_true_tensor, all_preds_tensor, average='micro', zero_division=0)
        val_recall_micro = recall_score(all_true_tensor, all_preds_tensor, average='micro', zero_division=0)
        val_f1_micro = f1_score(all_true_tensor, all_preds_tensor, average='micro', zero_division=0)

        precision_macro = precision_score(all_true_tensor, all_preds_tensor, average='macro', zero_division=0)
        recall_macro = recall_score(all_true_tensor, all_preds_tensor, average='macro', zero_division=0)
        f1_macro = f1_score(all_true_tensor, all_preds_tensor, average='macro', zero_division=0)

        # Print validation metrics
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Accuracy (Exact Match): {val_accuracy:.4f}")
        print(f"Validation Jaccard Similarity: {avg_jaccard:.4f}")
        print(f"Val Precision (micro): {val_precision_micro:.4f}")
        print(f"Val Recall (micro):    {val_recall_micro:.4f}")
        print(f"Val F1 Score (micro):  {val_f1_micro:.4f}")
        print(f"Val Precision (macro): {precision_macro:.4f}")
        print(f"Val Recall (macro):    {recall_macro:.4f}")
        print(f"Val F1 Score (macro):  {f1_macro:.4f}")
       

    # PLOT TRAINING AND VALIDATION METRICS
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(15, 5))

    # PLOT 1: TRAINING & VALIDATION LOSS
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_losses, label="Train Loss", color='blue')
    plt.plot(epochs_range, val_losses, label="Val Loss", color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)

    # PLOT 2: VALIDATION ACCURACY
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, val_accuracies, label="Exact Match", color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.grid(True)

    # PLOT 3: JACCARD SIMILARITY
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, val_jaccards, label="Jaccard", color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Similarity")
    plt.title("Jaccard Similarity")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_metrics.png"))
    plt.show()

    # SAVE MODEL AND TOKENIZER TO DIRECTORY
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    log_file.close()
    print(f"Validation predictions logged to: {log_file_path}")

    return model, test_loader


# Evaluate model on a test dataset
# model: trained HuggingFace model
# dataloader: PyTorch DataLoader containing test data
# mlb: MultiLabelBinarizer instance used to map labels to skill names
# threshold: probability threshold for binarizing predictions (default = 0.3)
def evaluate_model(model, dataloader, mlb, threshold=0.3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    y_prob = np.array(all_probs)
    y_true = np.array(all_labels)
    y_pred = (y_prob > threshold).astype(int)

    print("\nEvaluation Metrics:")
    print(f"Precision (micro): {precision_score(y_true, y_pred, average='micro', zero_division=0):.4f}")
    print(f"Recall (micro):    {recall_score(y_true, y_pred, average='micro', zero_division=0):.4f}")
    print(f"F1 Score (micro):  {f1_score(y_true, y_pred, average='micro', zero_division=0):.4f}")
    print(f"Precision (macro): {precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Recall (macro):    {recall_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"F1 Score (macro):  {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Exact Match:       {np.mean(np.all(y_true == y_pred, axis=1)):.4f}")
    print(f"Token Overlap:     {jaccard_similarity(y_true, y_pred, mlb.classes_):.4f}")

# Main function to orchestrate data loading, training, and evaluation
# csv_path: path to CSV file containing job summaries and skills
# model_dir: path to save trained model and artifacts (default = "./myModel")
# subset_fraction: float (0.0 - 1.0) indicating what fraction of the dataset to use (default = 1.0)
def main(csv_path, model_dir="./myModel", subset_fraction=1.0):
    summaries, skill_lists = load_data_from_csv(csv_path)

    # USE ONLY A FRACTION OF THE DATA
    if 0 < subset_fraction < 1.0:
        total_size = len(summaries)
        subset_size = int(total_size * subset_fraction)
        subset_indices = random.sample(range(total_size), subset_size)

        summaries = [summaries[i] for i in subset_indices]
        skill_lists = [skill_lists[i] for i in subset_indices]
        print(f"Using a subset of the data: {subset_size} out of {total_size} entries")

    # Encode labels
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(skill_lists)
    joblib.dump(mlb, "mlb.joblib")

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(mlb.classes_))

    dataset = JobSkillDataset(summaries, y, tokenizer)

    # Train model and get test data
    model, test_loader = train_model(dataset, model, tokenizer, mlb, model_dir, batch_size=16, epochs=5, threshold=0.2)

    # Evaluate on test data
    evaluate_model(model, test_loader, mlb, threshold=0.2)



# Parameters:
# csv_file: Path to the CSV file containing the dataset (e.g., "tech_industry_top500skills.csv")
# model_dir: Directory where the model and related outputs will be saved; 
#            generated using the base name of the CSV file
# subset_fraction: Fraction of the dataset to use (1.0 means use the full dataset)

# Execute script
if __name__ == "__main__":
    csv_file = "tech_industry_top500skills.csv"  
    model_dir = f"./myModel_{os.path.basename(csv_file).split('.')[0]}"
    main(csv_file, model_dir, subset_fraction=1.0)  


import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, EsmForSequenceClassification
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from sklearn.metrics import roc_auc_score, confusion_matrix

seed = 14
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

df = pd.read_csv('CEDAR_biological_activity.csv')


### Splitting data ###
train_df, val_df = train_test_split(df, test_size=0.20,stratify=df['Assay - Qualitative Measurement'])
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)


### tokenizing and preparing data for model loading ###
model_id = "facebook/esm2_t6_8M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_id)

def tokenize_function(examples):
    tokenized = tokenizer(
        examples["merged_sequence"],
        padding="max_length",
        truncation=True,
        max_length=1023
    )
    tokenized["labels"] = examples["Assay - Qualitative Measurement"]
    return tokenized

train_dataset = train_dataset.map(tokenize_function, batched=True) # applying tokenization
val_dataset = val_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"]) # setting format for pytorch
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8) # creating DataLoaders
eval_loader = DataLoader(val_dataset, batch_size=8)


### loading ESM-2 8M ###
model = EsmForSequenceClassification.from_pretrained(model_id, num_labels=2)


### configuring LoRA ###
lora_config = LoraConfig(
    task_type='SEQ_CLS',
    inference_mode=False,
    bias='lora_only',
    r = 5,
    lora_alpha=10,
    lora_dropout=0.2,
    target_modules=['dense'],
    modules_to_save=['out_proj']
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()


### initialising optimizer ###
optimizer = optim.AdamW(model.parameters(), lr= 0.00005, weight_decay=0.01)


### allocating model to cuda ###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


### training loop ###
def train(model, optimizer, dataloader, device):
    model.train()
    running_loss = 0.0
    correct = 0.0
    total = 0.0
    for batch in (dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        outputs = model(**batch) # forawrd pass
        loss = outputs.loss

        optimizer.zero_grad() # backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() # loss

        preds = torch.argmax(outputs.logits, dim=1) # accuracy
        correct += (preds == batch['labels']).sum().item()
        total += batch["labels"].size(0)
    
    avg_loss = running_loss / len(dataloader)
    accurracy = correct/total
    print(f"Training Loss: {avg_loss:.4f}, Training Accuracy: {accurracy:.4f}")


### evaluation loop ###
def evaluate(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    correct = 0.0
    total = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch) # forward pass
            loss = outputs.loss
            
            running_loss += loss.item() # loss

            probs = torch.nn.functional.softmax(outputs.logits, dim=1) 
            positive_probs = probs[:, 1].cpu().numpy()
                       
            preds = torch.argmax(outputs.logits, dim=1) # accuracy
            correct += (preds == batch['labels']).sum().item()
            total += batch['labels'].size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
            all_probs.extend(positive_probs)
    
    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total
    auc = roc_auc_score(all_labels, all_probs)

    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    print(f"Validation Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
    print(conf_matrix)


### running model ###
num_epochs = 16
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train(model, optimizer, train_dataloader, device)
    evaluate(model, eval_loader, device)

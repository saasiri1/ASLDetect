import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from utils.earlyStoping import EarlyStopping
import pickle
import torch.nn as nn
import time
def reset_weights(m):
    """Reset weights including Conv, Linear, and BatchNorm layers."""
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d)):
        m.reset_parameters()
import time
import wandb
from sklearn.metrics import precision_score, recall_score, f1_score

def train_fold_with_metrics(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs,
    patience,
    fold_id=None,
    use_wandb=True
):
    model.to(device)

    if use_wandb:
        wandb.init(
            project="sign-language-detection",
            name=f"ASLDetect-Fold-{fold_id}" if fold_id is not None else "ASLDetect",
            config={
                "epochs": num_epochs,
                "batch_size": train_loader.batch_size,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "fold": fold_id
            }
        )

    history = {k: [] for k in [
        'train_loss', 'val_loss', 'train_acc', 'val_acc',
        'train_recall', 'val_recall', 'train_precision', 'val_precision',
        'train_f1', 'val_f1'
    ]}

    early_stopping = EarlyStopping(patience=patience, min_delta=0.000001, mode='min', verbose=True)

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        train_labels, train_preds = [], []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            train_labels.extend(labels.cpu().numpy())
            train_preds.extend(preds.cpu().numpy())

        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_labels, val_preds = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(preds.cpu().numpy())

        # Metrics
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        train_recall = recall_score(train_labels, train_preds, average='macro', zero_division=0)
        val_recall = recall_score(val_labels, val_preds, average='macro', zero_division=0)

        train_precision = precision_score(train_labels, train_preds, average='macro', zero_division=0)
        val_precision = precision_score(val_labels, val_preds, average='macro', zero_division=0)

        train_f1 = f1_score(train_labels, train_preds, average='macro', zero_division=0)
        val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)

        # Save to history
        history['train_loss'].append(train_loss / train_total)
        history['val_loss'].append(val_loss / val_total)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_recall'].append(train_recall)
        history['val_recall'].append(val_recall)
        history['train_precision'].append(train_precision)
        history['val_precision'].append(val_precision)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)

        # Logging
        print(
            f"[Fold {fold_id}] Epoch {epoch+1}/{num_epochs} | "
            f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f} | "
            f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f} | "
            f"Time: {time.time() - epoch_start:.2f}s"
        )

        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss / train_total,
                "val_loss": val_loss / val_total,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "train_f1": train_f1,
                "val_f1": val_f1,
                "train_recall": train_recall,
                "val_recall": val_recall,
                "train_precision": train_precision,
                "val_precision": val_precision
            })

        # Early stopping
        early_stopping(val_loss / val_total, model)
        if early_stopping.early_stop:
            print("✅ Early stopping triggered.")
            break

    if use_wandb:
        wandb.finish()

    return history



def train_final_model(model, full_dataset, criterion, optimizer, device, num_epochs, patience, filename):
    """
    Train the final model on the full training dataset after k-fold.
    Returns the trained model and its history.
    """
    model.apply(reset_weights)
    model.to(device)

    full_loader = DataLoader(full_dataset, batch_size=64, shuffle=True)

    early_stopping = EarlyStopping(patience=patience, min_delta=0.000001, mode='min', verbose=True)

    history = {'train_loss': [], 'train_acc': []}

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in full_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(full_loader)
        epoch_acc = correct / total

        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)

        print(f"Epoch {epoch+1}/{num_epochs} - Final Model Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        early_stopping(epoch_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    torch.save(model.state_dict(), f'{filename}_final_model.pth')
    print(f"Final model saved to '{filename}_final_model.pth'")

    return model, history


def k_fold_cross_validation(model, dataset, device, num_epochs, patience, n_splits=5, batch_size=32, filename='kfold_results'):
    """
    Run k-fold cross-validation AND train the final model at the end.
    Returns the final trained model and history.
    """
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    

    all_fold_histories = []
    fold_accuracies = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\n=== Starting Fold {fold+1}/{n_splits} ===")

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True,num_workers=8, pin_memory=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False,num_workers=8, pin_memory=True)

        model.apply(reset_weights)
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()

        history = train_fold_with_metrics(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, patience)
        best_val_acc = max(history['val_acc'])
        with open(f"{filename}_fold_{fold}_best_val_acc.txt", 'w') as f:
            f.write(str(best_val_acc))
        fold_accuracies.append(best_val_acc)
        all_fold_histories.append(history)
        with open(f"{filename}_fold_{fold}_history.pkl", 'wb') as f:
            pickle.dump(history, f)

    plot_all_folds_metrics(all_fold_histories, filename)

    print("\n=== Training Final Model on Full Dataset ===")

    final_model, final_history = train_final_model(
        model, dataset, criterion, optimizer, device, num_epochs, patience, filename
    )

    return final_model, final_history , fold_accuracies


def plot_all_folds_metrics(all_fold_histories, filename):
    metrics = ['loss', 'acc', 'recall', 'precision', 'f1']
    metric_names = {
        'loss': 'Loss',
        'acc': 'Accuracy',
        'recall': 'Recall',
        'precision': 'Precision',
        'f1': 'F1 Score'
    }

    for metric in metrics:
        plt.figure(figsize=(14, 6))

        for fold, history in enumerate(all_fold_histories):
            plt.plot(history[f'train_{metric}'], label=f'Fold {fold+1} - Train {metric_names[metric]}', linestyle='--')
            plt.plot(history[f'val_{metric}'], label=f'Fold {fold+1} - Val {metric_names[metric]}')

        plt.title(f'{metric_names[metric]} Across All Folds')
        plt.xlabel('Epoch')
        plt.ylabel(metric_names[metric])
        plt.legend()
        plt.grid(True)
        plt.savefig(f"figures/  {filename}_all_folds_{metric}.png")

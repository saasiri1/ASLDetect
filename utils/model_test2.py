import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def test(model, test_loader, criterion, device):
    # Test the model and evaluate
    total_test, correct_test, total_loss = 0, 0, 0
    all_labels, all_predictions = [], []
    print('Testing the model...')
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)  # Multiply by batch size

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    print('Testing complete.')
    print('Calculating test metrics...')
    
    # Calculate test metrics
    test_acc = 100 * correct_test / total_test
    test_loss = total_loss / total_test  # Average loss per sample
    return test_loss, test_acc, all_labels, all_predictions


def modelTest(test_loader, test_path, models, device, filename, learning_rate, train_dataset, num_epochs, criterions, model_names):
    with open(f'{filename}.csv', "w", newline='') as csvfile:
        fieldnames = ['Model', 'Dataset', 'Learning Rate', 'Epochs', 'Test Loss', 'Test Accuracy', 'Test Recall', 'Test Precision', 'Test F1']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for idx, model in enumerate(models):  # Iterate over models
            if isinstance(model, torch.nn.Module):
                # Test the model
                test_loss, test_acc, all_labels, all_predictions = test(model, test_loader, criterions[idx], device)

                # Calculate additional metrics
                test_recall = recall_score(all_labels, all_predictions, average='macro', zero_division=1)
                test_precision = precision_score(all_labels, all_predictions, average='macro', zero_division=1)
                test_f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=1)
                
                # Classification report and confusion matrix
                class_report = classification_report(all_labels, all_predictions, target_names=train_dataset.classes)
                cm = confusion_matrix(all_labels, all_predictions)

                # Save results to CSV
                writer.writerow({
                    'Model': model_names[idx],
                    'Dataset': test_path,
                    'Learning Rate': learning_rate,
                    'Epochs': num_epochs[idx],
                    'Test Loss': f'{test_loss:.4f}',
                    'Test Accuracy': f'{test_acc:.2f}%',
                    'Test Recall': f'{test_recall:.2f}',
                    'Test Precision': f'{test_precision:.2f}',
                    'Test F1': f'{test_f1:.2f}'
                })

                # Save classification report in a text file
                with open(f'{filename}_{model_names[idx]}_details.txt', 'w') as file:
                    file.write(f'Model: {model_names[idx]}\n')
                    file.write(f'Test Result for {test_path}:\n')
                    file.write(f'Learning Rate: {learning_rate}\n')
                    file.write(f'Epochs: {num_epochs[idx]}\n')
                    file.write(f'Test Loss: {test_loss:.4f}\n')
                    file.write(f'Test Accuracy: {test_acc:.2f}%\n')
                    file.write(f'Test Recall: {test_recall:.2f}\n')
                    file.write(f'Test Precision: {test_precision:.2f}\n')
                    file.write(f'Test F1: {test_f1:.2f}\n')
                    file.write("\nClassification Report:\n")
                    file.write(class_report)
                # Save confusion matrix as a figure
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
                plt.title(f'Confusion Matrix - {model_names[idx]}')
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                # Save the figure
                plt.savefig(f'{filename}_{model_names[idx]}_confusion_matrix.png')
                plt.close()  # Close the plot to free memory

                print(f"Confusion matrix saved as {filename}_{model_names[idx]}_confusion_matrix.png")

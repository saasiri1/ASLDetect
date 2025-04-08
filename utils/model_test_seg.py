import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

def test(model, test_loader, criterion, device):
     # Test the model and evaluate
    total_test, correct_test, total_loss = 0, 0, 0
    all_labels, all_predictions = [], []
    print('Testing the model...')
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    print('Testing complete.')
    print('Calculating test metrics...')
    # Calculate test metrics
    test_acc = 100 * correct_test / total_test
    test_loss = total_loss / len(test_loader.dataset)
    return test_loss, test_acc, all_labels, all_predictions


def modelTest(test_loader,test_path,model,device,filename,learning_rate ,train_dataset,num_epochs,criterion,modelName):
    test_loss, test_acc, all_labels, all_predictions = test(model, test_loader, criterion, device)
    test_recall = recall_score(all_labels, all_predictions, average='macro', zero_division=1)
    test_precision = precision_score(all_labels, all_predictions, average='macro', zero_division=1)
    test_f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=1)

    # Generate the classification report
    class_report = classification_report(all_labels, all_predictions, target_names=train_dataset.classes)
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    print("saving results")
    # Save results to the result file
    with open(f'{filename}.txt', "w") as file:
        file.write(f'Model:'+modelName+'Test Result\n')
        file.write(f'Dataset: {test_path}\n')
        file.write(f'Learning Rate: {learning_rate}\n')
        file.write(f'Epochs: {num_epochs}\n')
        file.write(f'Test Loss: {test_loss:.4f}\n')
        file.write(f'Test Acc: {test_acc:.2f}%\n')
        file.write(f'Test Recall: {test_recall:.2f}\n')
        file.write(f'Test Precision: {test_precision:.2f}\n')
        file.write(f'Test F1: {test_f1:.2f}\n')
        file.write("\nClassification Report:\n")
        file.write(class_report)
        file.write("\nConfusion Matrix:\n")
        file.write(str(cm))

    # Display the classification report on the screen
    print("\nClassification Report:\n")
    print(class_report)

    print(f'Results and classification report saved to {filename}')
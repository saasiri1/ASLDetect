import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def test(model, test_loader, criterion, device):
    """
    Test the model and evaluate its performance.

    Args:
        model: Trained model to test.
        test_loader: DataLoader for the test set.
        criterion: Loss function.
        device: Device to run the model on (e.g., 'cuda' or 'cpu').

    Returns:
        test_loss: Average test loss.
        test_acc: Test accuracy.
        all_labels: List of true labels.
        all_predictions: List of predicted labels.
    """
    model.eval()  # Set the model to evaluation mode
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
    test_acc = 100 * correct_test / total_test
    test_loss = total_loss / len(test_loader)
    return test_loss, test_acc, all_labels, all_predictions


def plot_confusion_matrix(cm, class_names, filename):
    """
    Plot and save the confusion matrix.

    Args:
        cm: Confusion matrix.
        class_names: List of class names.
        filename: Filename to save the plot.
    """
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(filename)
    plt.close()


def save_results(filename, model_name, test_path, test_loss, test_acc, test_recall, test_precision, test_f1, class_report, cm):
    """
    Save test results to a file.

    Args:
        filename: Filename to save the results.
        model_name: Name of the model.
        test_path: Path to the test dataset.
        test_loss: Test loss.
        test_acc: Test accuracy.
        test_recall: Test recall.
        test_precision: Test precision.
        test_f1: Test F1 score.
        class_report: Classification report.
        cm: Confusion matrix.
    """
    with open(filename, "w") as file:
        file.write(f'Model: {model_name} Test Result\n')
        file.write(f'Dataset: {test_path}\n')
        file.write(f'Test Loss: {test_loss:.4f}\n')
        file.write(f'Test Acc: {test_acc:.2f}%\n')
        file.write(f'Test Recall: {test_recall:.2f}\n')
        file.write(f'Test Precision: {test_precision:.2f}\n')
        file.write(f'Test F1: {test_f1:.2f}\n')
        file.write("\nClassification Report:\n")
        file.write(class_report)
        file.write("\nConfusion Matrix:\n")
        file.write(str(cm))


def modelTest(test_loader, test_path, model, device, filename, model_name, class_names):
    """
    Test the model, calculate metrics, and save results.

    Args:
        test_loader: DataLoader for the test set.
        test_path: Path to the test dataset.
        model: Trained model to test.
        device: Device to run the model on (e.g., 'cuda' or 'cpu').
        filename: Base filename to save results and plots.
        criterion: Loss function.
        model_name: Name of the model.
        class_names: List of class names.
    """
    # Test the model
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, all_labels, all_predictions = test(model, test_loader, criterion, device)

    # Calculate additional metrics
    test_recall = recall_score(all_labels, all_predictions, average='macro', zero_division=1)
    test_precision = precision_score(all_labels, all_predictions, average='macro', zero_division=1)
    test_f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=1)

    # Generate classification report and confusion matrix
    class_report = classification_report(all_labels, all_predictions, target_names=class_names)
    cm = confusion_matrix(all_labels, all_predictions)

    # Plot and save confusion matrix
    plot_confusion_matrix(cm, class_names, f'{filename}_confusion_matrix.png')

    # Save results to a file
    save_results(
        f'{filename}.txt', model_name, test_path, test_loss, test_acc, 
        test_recall, test_precision, test_f1, class_report, cm
    )

    # Display the classification report
    print("\nClassification Report:\n")
    print(class_report)

    print(f'Results and classification report saved to {filename}')

import torch
import torch.nn as nn
from utils.earlyStoping import EarlyStopping
from utils.plot_model import plot_training_performance
import torch.optim as optim


def freeze_pretrained_layers(model):
    # Freeze ResNet layers (the encoder)
    for param in model.base_model.parameters():
        param.requires_grad = False

def unfreeze_pretrained_layers(model):
    # Unfreeze ResNet layers (the encoder)
    for param in model.base_model.parameters():
        param.requires_grad = True

def train_model(model, train_loader, val_loader, device, num_epochs=25, patience=15, gamma=0.1, filename='model', freeze_epochs=5):
    print("Training the model...")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    # Scheduler to reduce LR when validation loss stops improving
    scheduler_loss = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True, threshold=0.0001, cooldown=2, min_lr=1e-6
    )

    # Scheduler to reduce LR when validation accuracy stops improving
    scheduler_acc = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=5, verbose=True, threshold=0.001, cooldown=2, min_lr=1e-6
    )
    # Freeze the pre-trained layers initially
    freeze_pretrained_layers(model)

    # Lists to store loss and accuracy for plotting
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    early_stopping = EarlyStopping(patience=patience, min_delta=0.000001, mode='min', verbose=True)

    for epoch in range(num_epochs):
        if epoch == freeze_epochs:
            # Unfreeze the pre-trained layers after freeze_epochs epochs
            unfreeze_pretrained_layers(model)
            optimizer = optim.AdamW(model.parameters(), lr=1e-4)  # Lower learning rate after unfreezing

        model.train()
        running_loss, running_correct = 0.0, 0.0

        # Training phase
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            running_correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        train_acc = 100. * running_correct / len(train_loader.dataset)
        train_loss_history.append(epoch_loss)
        train_acc_history.append(train_acc)

        print(f'Epoch {epoch}/{num_epochs-1}, Train Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.2f}%')

        # Validation phase
        model.eval()
        val_loss, val_correct = 0.0, 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100. * val_correct / len(val_loader.dataset)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        print(f'Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.2f}%')

        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        # Step the scheduler based on validation loss
        scheduler_loss.step(val_loss)
        # Step the scheduler based on validation accuracy
        scheduler_acc.step(val_acc)
        # Print the current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Current learning rate: {current_lr}')  
    plot_training_performance(train_loss_history, val_loss_history, train_acc_history, val_acc_history, num_epochs=epoch+1, plot_title=f'{filename}_training_perf.png')

    return model,(epoch+1)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = nn.CrossEntropyLoss()(inputs, targets)
        else:
            BCE_loss = nn.CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss
    

def stateOfArtsTraining(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs=25,
    patience=15,
    filename='model'
):
    # to device
    print("Training the model...")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    # Single scheduler for learning rate adjustment
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, 
        verbose=True, threshold=0.0001, cooldown=2, min_lr=1e-6
    )

    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=patience, 
        min_delta=0.000001, 
        mode='min', 
        verbose=True
    )

    # History tracking
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'lr': []
    }

    best_acc = 0.0  # Track best validation accuracy

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss, running_correct = 0.0, 0.0
        
        # Training phase
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_correct += torch.sum(preds == labels.data)

        # Calculate training metrics
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = running_correct.double() / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss, val_correct = 0.0, 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)

        # Calculate validation metrics
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = val_correct.double() / len(val_loader.dataset)
        
        # Update learning rate scheduler
        scheduler.step(epoch_val_loss)
        
        # Track metrics
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Print metrics
        print(f"Train Loss: {epoch_train_loss:.4f} | Acc: {100*epoch_train_acc:.2f}%")
        print(f"Val Loss: {epoch_val_loss:.4f} | Acc: {100*epoch_val_acc:.2f}%")
        print(f"LR: {history['lr'][-1]:.2e}")

        # Check for best model
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            torch.save(model.state_dict(), f"{filename}_best.pth")
            print("Saved new best model")

        # Early stopping check
        early_stopping(epoch_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # Load best model weights
    model.load_state_dict(torch.load(f"{filename}_best.pth"))
    
    # # Plot training curves
    # plot_training_performance(
    #     history['train_loss'],
    #     history['val_loss'],
    #     history['train_acc'],
    #     history['val_acc'],
    #     num_epochs=epoch+1,
    #     plot_title=f'{filename}_training_perf.png'
    # )

    return model, epoch+1
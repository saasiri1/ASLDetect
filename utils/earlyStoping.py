import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, mode='min', verbose=False, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights
        self.best_metric = None
        self.best_epoch = None
        self.epochs_since_improvement = 0
        self.stopped_epoch = 0
        self.early_stop = False
        self.best_weights = None

    def __call__(self, current_metric, model):
        try:
            if self.best_metric is None:
                self.best_metric = current_metric
                self.best_weights = model.state_dict()
                self.best_epoch = self.stopped_epoch
            elif self._is_improvement(current_metric):
                self.best_metric = current_metric
                self.best_weights = model.state_dict()
                self.best_epoch = self.stopped_epoch
                self.epochs_since_improvement = 0
                if self.verbose:
                    print(f'Validation metric improved to {current_metric:.4f} from {self.best_metric:.4f}')
            else:
                self.epochs_since_improvement += 1
                if self.epochs_since_improvement >= self.patience:
                    self.early_stop = True
                    if self.restore_best_weights:
                        model.load_state_dict(self.best_weights)
                    if self.verbose:
                        print(f'Early stopping triggered. Restoring model weights from epoch {self.best_epoch}.')
        except Exception as e:
            print(f"EarlyStopping encountered an error: {str(e)}")

        self.stopped_epoch += 1

    def _is_improvement(self, current_metric):
        if self.mode == 'min':
            return current_metric <= self.best_metric - self.min_delta
        else:  # mode == 'max'
            return current_metric >= self.best_metric + self.min_delta

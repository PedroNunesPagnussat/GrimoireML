from typing import List, Optional

class History:
    """
    History class for storing and managing loss and metrics data during model training.
    
    The History object gets returned by the `fit` method of models. It holds a record of 
    the training loss values and metrics values at successive epochs, as well as validation 
    loss values and validation metrics values (if applicable).
    
    Attributes:
        history (dict): A dictionary containing training metrics, loss, and optionally, validation metrics and loss.
            
    Methods:
        append_epoch(loss: float, metrics: dict)
            Appends the loss and metrics for a training epoch.
        append_validation(val_loss: float, val_metrics: dict)
            Appends the validation loss and metrics for a training epoch.
    """
    
    def __init__(self, has_validation: bool = False, metric_names: Optional[List[str]] = None):
        self._has_validation = has_validation
        self.history = {'loss': []}
        
        if metric_names:
            for name in metric_names:
                self.history[name] = []
        
        if self._has_validation:
            self.history['val_loss'] = []
            if metric_names:
                for name in metric_names:
                    self.history[f'val_{name}'] = []


    def append_epoch(self, loss: float, metrics: dict) -> None:
        """
        Append the metrics and loss for the current epoch to the history.
        
        Args:
            loss (float): The loss for the current epoch.
            metrics (dict): A dictionary containing the metrics for the current epoch.
        """
        self.history['loss'].append(loss)
        for name, val in metrics.items():
            self.history[name].append(val)

    def append_validation(self, val_loss: float, val_metrics: dict) -> None:
        """
        Append the validation metrics and loss for the current epoch to the history.
        
        Args:
            val_loss (float): The validation loss for the current epoch.
            val_metrics (dict): A dictionary containing the validation metrics for the current epoch.
        """
        self.history['val_loss'].append(val_loss)
        for name, val in val_metrics.items():
            self.history[f'val_{name}'].append(val)

            

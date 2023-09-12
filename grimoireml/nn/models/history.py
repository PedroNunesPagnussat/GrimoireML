class History:
    """
    History class for storing and managing loss and metrics data during model training.
    
    The History object gets returned by the `fit` method of models. It holds a record of 
    the training loss values and metrics values at successive epochs, as well as validation 
    loss values and validation metrics values (if applicable).
    
    Attributes:
        history (dict): A dictionary containing training metrics, loss, and optionally, validation metrics and loss.
            - 'loss': List of float values indicating the loss over epochs.
            - 'metric': List of float values indicating the metric over epochs.
            - 'val_loss': List of float values indicating the validation loss over epochs (if validation data provided).
            - 'val_metric': List of float values indicating the validation metric over epochs (if validation data provided).
            
    Methods:
        append(loss: float, metric: float, val_loss: float = None, val_metric: float = None)
            Appends the loss and metrics for a training epoch, as well as optional validation loss and metrics.
    """
    
    def __init__(self, has_validation: bool = False):
        self._has_validation = has_validation

        if self._has_validation:
            self.history = {'loss': [], 'val_loss': [], 'metric': [], 'val_metric': []}
        else:
            self.history = {'loss': [], 'metric': []}

    def append_epoch(self, loss: float, metric: float) -> None:
        """
        Append the metrics and loss for the current epoch to the history.
        
        Args:
            loss (float): The loss for the current epoch.
            metric (float): The metric for the current epoch.
        """
        self.history['loss'].append(loss)
        self.history['metric'].append(metric)
        

    def append_validation(self, val_loss: float, val_metric: float) -> None:
        """
        Append the validation metrics and loss for the current epoch to the history.
        
        Args:
            val_loss (float): The validation loss for the current epoch.
            val_metric (float): The validation metric for the current epoch.
        """
        if self._has_validation:
            self.history['val_loss'].append(val_loss)
            self.history['val_metric'].append(val_metric)


            

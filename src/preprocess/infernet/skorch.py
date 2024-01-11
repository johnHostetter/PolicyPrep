import torch
from skorch.utils import to_tensor
from skorch import NeuralNetRegressor


class InferNet(NeuralNetRegressor):
    def __init__(self, module, *args, criterion=torch.nn.MSELoss, **kwargs):
        super(InferNet, self).__init__(module, *args, criterion=criterion, **kwargs)

    def get_loss(self, y_pred, y_true, X=None, training=False):
        y_true = to_tensor(y_true, device=self.device)
        return self.criterion_(
            y_pred.sum(1).flatten().to(self.device), y_true.sum(dim=1)
        )

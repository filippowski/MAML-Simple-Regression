from trainer import *


class MetaLearningMSELoss(nn.Module):

    def __init__(self, reduction='mean', verbose=False):
        super(MetaLearningMSELoss, self).__init__()
        self.lf = MSELoss(reduction=reduction)
        self.verbose = verbose

    def forward(self, predict, target, task_predict):
        """Replaces meta-model predictions with task model predictions
        and after that regular MSE loss is calculated during forward pass.
        During backward pass gradients of test loss from a task-model
        w.r.t. initial weights of meta-model will be calculated.
        """
        if self.verbose: print("initial test loss:", self.lf(task_predict, target))
        if self.verbose: print("predict before replace:", predict)
        if self.verbose: print("meta loss before replace:", self.lf(predict, target))

        # replace meta model prediction with task model prediction
        predict.data = task_predict.data

        # calculate loss
        loss = self.lf(predict, target)

        if self.verbose: print("predict after replace:", predict)
        if self.verbose: print("meta loss after replace:", self.lf(predict, target), loss)

        # check the last bias gradient
        if self.verbose: print("bias grad:", 2 * (predict - target).mean())

        return loss

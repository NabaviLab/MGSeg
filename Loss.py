import torch
import torch.nn as nn
import torch.nn.functional as F

class CompositeLoss(nn.Module):
    """ Composite Loss Function: Dice + CE + L2 Regularization """
    def __init__(self, lambda_reg=0.001, beta_start=0.3, beta_end=2.0, total_epochs=100, t0=10):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.total_epochs = total_epochs
        self.t0 = t0

    def dice_loss(self, pred, target):
        smooth = 1.0
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))

    def cross_entropy_loss(self, pred, target):
        return F.binary_cross_entropy_with_logits(pred, target)

    def l2_regularization(self, model):
        reg_loss = 0.0
        for param in model.parameters():
            reg_loss += torch.norm(param, 2)
        return reg_loss

    def forward(self, pred, target, model, t):
        beta = self.beta_start + (self.beta_end - self.beta_start) * (t / self.total_epochs)
        alpha = 1 / (1 + torch.exp(-beta * (t - self.t0)))
        alpha1 = 2 * alpha / 3
        alpha2 = alpha / 3
        
        dice = self.dice_loss(pred, target)
        ce = self.cross_entropy_loss(pred, target)
        reg = self.l2_regularization(model)
        
        loss = alpha1 * dice + alpha2 * ce + (1 - alpha) * self.lambda_reg * reg
        return loss
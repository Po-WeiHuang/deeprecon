import torch
from torch import nn
import torch.nn.functional as F




class ResolutionLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, predict: dict, truth: dict):
        # MSE do normalisation to the tensor shape
        positionloss = F.mse_loss(predict["position"], truth["position"]) / 3
        energyloss = F.mse_loss(predict["energy"], truth["energy"])
        evtloss = F.mse_loss(predict["evtime"], truth["evtime"]) 
        return positionloss + energyloss + evtloss , positionloss, energyloss, evtloss
        #return  energyloss   , positionloss, energyloss, evtloss

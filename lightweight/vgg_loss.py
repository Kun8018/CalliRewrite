import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class VGG16PerceptualLoss(nn.Module):
    def __init__(self, layers=['relu1_2', 'relu2_2', 'relu3_3', 'relu5_1']):
        super().__init__()
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.layers = layers

        self.layer_names = {
            'relu1_2': 3,
            'relu2_2': 8,
            'relu3_3': 15,
            'relu5_1': 29
        }

        self.vgg = nn.ModuleList()
        last_idx = 0
        for layer in layers:
            idx = self.layer_names[layer]
            self.vgg.append(vgg16[last_idx:idx+1])
            last_idx = idx + 1

        for param in self.vgg.parameters():
            param.requires_grad = False

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        pred_rgb = torch.cat([pred, pred, pred], dim=1)
        target_rgb = torch.cat([target, target, target], dim=1)

        pred_rgb = (pred_rgb - self.mean) / self.std
        target_rgb = (target_rgb - self.mean) / self.std

        losses = []
        x_pred, x_target = pred_rgb, target_rgb
        for layer in self.vgg:
            x_pred = layer(x_pred)
            x_target = layer(x_target)
            loss = F.l1_loss(x_pred, x_target)
            losses.append(loss)

        return sum(losses) / len(losses)

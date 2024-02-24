import torch
import torch.nn as nn
from iou import iou

class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

    def forward(self, pred, target, anchors):
        obj = target[..., 0] == 1
        no_obj = target[..., 0] == 0

        no_object_loss = self.bce(pred[..., 0:1][no_obj], target[..., 0:1][no_obj])

        anchors = anchors.reshape(1, 3, 1, 1, 2)
        box_preds = torch.cat([self.sigmoid(pred[..., 1:3]), torch.exp(pred[..., 3:5]) * anchors], dim=-1)
        ious = iou(box_preds[obj], target[..., 1:5][obj]).detach()
        object_loss = self.mse(self.sigmoid(pred[..., 0:1][obj]), ious * target[..., 0:1][obj])

        pred[..., 1:3] = self.sigmoid(pred[..., 1:3])
        target[..., 3:5] = torch.log(1e-6 + target[..., 3:5] / anchors)
        box_loss = self.mse(pred[..., 1:5][obj], target[..., 1:5][obj])

        class_loss = self.cross_entropy(pred[..., 5:][obj], target[..., 5][obj].long())

        total_loss = box_loss + object_loss + no_object_loss + class_loss
        return total_loss

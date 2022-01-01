import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MOCO(nn.Module):
    def __init__(self, feature_dim=128,moco_train=False):
        super(MOCO, self).__init__()
        self.moco_train = moco_train
        self.backbone = models.resnet50(pretrained=False)
        # todo: delete mlp head and use only backbone, no identiry
        """
        if moco_train:
            self.backbone.fc = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                                      nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))        
        """
        self.backbone.fc = nn.Identity()
        # MLP head
        self.mlp_head = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                                      nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, start_dim=1)
        out = self.mlp_head(x)
        return F.normalize(x, dim=-1), F.normalize(out, dim=-1)

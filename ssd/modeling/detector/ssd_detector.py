from torch import nn

from ssd.modeling.backbone import build_backbone
from ssd.modeling.box_head import build_box_head

import matplotlib.pyplot as plt


class SSDDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.box_head = build_box_head(cfg)

    def forward(self, images, targets=None):
        features,mask_p = self.backbone(images)
        detections, detector_losses = self.box_head(features, mask_p, targets)
        # print("len detections",len(detections),"detections == ",detections[0])
        # plt.figure()
        # plt.imshow(images[0][0].cpu().detach().numpy())
        # plt.show()
        if self.training:
            return detector_losses
        return detections

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_HAT_ROOT = os.path.dirname(_THIS_DIR)
_PROJECT_ROOT = os.path.dirname(_HAT_ROOT)
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

from lib.Network_Res2Net_GRA_NCD import Network  # noqa: E402


class CamouflagePerceptionModule(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        model_path=None,
        frozen=True,
    ):
        super().__init__()
        if model_path is None:
            model_path = os.path.join(
                _HAT_ROOT, "pretrained_models", "sinetv2", "Net_epoch_best.pth"
            )

        print(f"Loading SINet-V2 weights from: {model_path}")
        self.cod_model = Network(channel=32, imagenet_pretrained=False)
        state_dict = torch.load(model_path, map_location="cpu")
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        msg = self.cod_model.load_state_dict(state_dict, strict=False)
        print(
            "SINet-V2 weights loaded. "
            f"Missing keys: {len(msg.missing_keys)}, unexpected keys: {len(msg.unexpected_keys)}"
        )

        if frozen:
            self.cod_model.eval()
            for param in self.cod_model.parameters():
                param.requires_grad = False

        self.adapter_conv = nn.Conv2d(2048, hidden_dim, kernel_size=1)
        self.uncertainty_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        with torch.no_grad():
            feat = self.cod_model.resnet.conv1(x)
            feat = self.cod_model.resnet.bn1(feat)
            feat = self.cod_model.resnet.relu(feat)
            feat = self.cod_model.resnet.maxpool(feat)
            feat = self.cod_model.resnet.layer1(feat)
            feat = self.cod_model.resnet.layer2(feat)
            feat = self.cod_model.resnet.layer3(feat)
            feat = self.cod_model.resnet.layer4(feat)

        camo_emb = self.adapter_conv(feat)
        uncertainty_map = self.uncertainty_proj(camo_emb)
        return camo_emb, uncertainty_map

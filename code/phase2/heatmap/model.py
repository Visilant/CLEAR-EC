"""ConvNeXt-V2-Nano encoder + light FPN decoder -> full-resolution 1-channel heatmap."""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

ENCODER = "convnextv2_nano.fcmae_ft_in22k_in1k"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def conv_block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
                         nn.GELU(), nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.GELU())


class HeatmapNet(nn.Module):
    def __init__(self, pretrained=True, width=96, stem_width=32):
        super().__init__()
        self.encoder = timm.create_model(ENCODER, pretrained=pretrained, features_only=True,
                                         out_indices=(0, 1, 2, 3), in_chans=3)
        chans = self.encoder.feature_info.channels()  # strides 4, 8, 16, 32
        self.lateral = nn.ModuleList([nn.Conv2d(c, width, 1) for c in chans])
        self.smooth = nn.ModuleList([conv_block(width, width) for _ in chans[:-1]])
        self.stem = conv_block(3, stem_width)  # full-res shallow features
        self.head = nn.Sequential(conv_block(width + stem_width, stem_width * 2),
                                  nn.Conv2d(stem_width * 2, 1, 1))
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        """x: (B,1,H,W) float in [0,1], H and W multiples of 32. Returns logits (B,1,H,W)."""
        x3 = (x.expand(-1, 3, -1, -1) - self.mean) / self.std
        feats = self.encoder(x3)
        p = self.lateral[-1](feats[-1])
        for i in range(len(feats) - 2, -1, -1):
            p = F.interpolate(p, size=feats[i].shape[-2:], mode="bilinear", align_corners=False)
            p = self.smooth[i](p + self.lateral[i](feats[i]))
        p = F.interpolate(p, size=x.shape[-2:], mode="bilinear", align_corners=False)
        s = self.stem(x3)
        return self.head(torch.cat([p, s], dim=1))


def masked_mse(logits, target, mask, pos_weight=1.0):
    """Mean squared error between sigmoid(logits) and target inside mask; optional up-weighting of peaks."""
    pred = torch.sigmoid(logits.float())
    w = mask * (1.0 + (pos_weight - 1.0) * target)
    return ((pred - target) ** 2 * w).sum() / w.sum().clamp_min(1.0)


def pad_to_multiple(img, m=32):
    """Pad a (B,1,H,W) tensor (reflect) so H and W are multiples of m; returns padded, (H,W)."""
    H, W = img.shape[-2:]
    ph, pw = (-H) % m, (-W) % m
    if ph or pw:
        img = F.pad(img, (0, pw, 0, ph), mode="reflect")
    return img, (H, W)


@torch.no_grad()
def predict_heatmap(model, frame_u8, device, amp=True):
    """Full-frame heatmap (H,W) float32 in [0,1] for a uint8 grey image."""
    x = torch.from_numpy(frame_u8.astype("float32") / 255.0)[None, None].to(device)
    x, (H, W) = pad_to_multiple(x)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
        logits = model(x)
    return torch.sigmoid(logits.float())[0, 0, :H, :W].cpu().numpy()

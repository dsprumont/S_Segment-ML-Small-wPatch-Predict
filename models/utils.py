import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Conv2dWithNorm', 'UpSample2d']


class Conv2dWithNorm(nn.Conv2d):
    """
    Conv2d module that also applies given normalization module.
    """
    def __init__(self, *args, **kargs):
        norm = kargs.pop('norm', None)
        super().__init__(*args, **kargs)
        self.norm = norm

    def forward(self, x):
        h = super().forward(x)
        if self.norm is not None:
            h = self.norm(h)
        return h


class UpSample2d(nn.Module):
    """
    UpSample2d module that upscale inputs to a given size.
    """
    def __init__(self, size, mode='bilinear', align_corners=True):
        super(UpSample2d, self).__init__()
        self.size = size
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(x,
                             size=self.size,
                             mode=self.mode,
                             align_corners=self.align_corners)

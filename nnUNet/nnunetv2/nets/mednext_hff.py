"""
MedNeXtHFF — MedNeXt backbone with HFF-style high-frequency dual-route processing.

LF route: raw input → MedNeXt backbone
HF route: Laplacian edge-enhanced input → same MedNeXt backbone (shared weights)
Output: element-wise average of both routes.

No FFT, no uncertainty sampling — just dual processing with edge enhancement.
This avoids all numerical instability of the original HFF (NaN from fp16
softmax, Cholesky in Half, etc.) while retaining the high-frequency signal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MedNeXtHFF(nn.Module):
    def __init__(self, mednext_model):
        super().__init__()
        self.mednext = mednext_model
        self.do_ds = getattr(mednext_model, 'do_ds', False)

        # Learnable fusion weight: output = α * LF + (1-α) * HF
        self.logit_alpha = nn.Parameter(torch.zeros(1))

        # 3D Laplacian kernel (fixed, no grad)
        laplacian = torch.tensor(
            [[[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
              [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
              [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]],
            dtype=torch.float32
        ).unsqueeze(0)  # [1, 1, 3, 3, 3]
        self.register_buffer('laplacian_kernel', laplacian, persistent=False)

    def apply_laplacian(self, x):
        c = x.shape[1]
        kernel = self.laplacian_kernel.repeat(c, 1, 1, 1, 1)
        return F.conv3d(x, kernel, padding=1, groups=c)

    def alpha(self):
        return torch.sigmoid(self.logit_alpha)

    def fuse(self, lf, hf):
        a = self.alpha()
        return a * lf + (1 - a) * hf

    def forward(self, x):
        lf = self.mednext(x)
        x_edge = self.apply_laplacian(x)
        hf = self.mednext(x_edge)

        if self.do_ds and isinstance(lf, (list, tuple)):
            return [self.fuse(l, h) for l, h in zip(lf, hf)]
        else:
            return self.fuse(lf, hf)

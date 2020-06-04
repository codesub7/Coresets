import numpy as np

import torch.nn as nn

def broadcast_to_shape(z, shape):
    if isinstance(z, (int, float)):
        return np.broadcast_to(z, shape).astype(float).tolist()

    z = np.array(z)

    while z.shape != shape[:len(z.shape)]:
        z = z[np.newaxis, :]

    while len(z.shape) != len(shape):
        z = z[:, np.newaxis]

    z = np.broadcast_to(z, shape)
    assert z.shape == shape

    z = np.ascontiguousarray(z)

    return z


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

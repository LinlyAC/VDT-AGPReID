import numpy as np
import torch
import pdb

out = torch.nn.Parameter(torch.zeros(256, 768))
n_pos, dim = out.shape
position_enc = np.array(
    [
        [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
        for pos in range(n_pos)
    ]
)
out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
out.detach_()

x1 = torch.tensor([1,2,3])
x2 = torch.tensor([4,5,6])
y = torch.stack([x1, x2], dim=-1).flatten(-2,-1)
pdb.set_trace()
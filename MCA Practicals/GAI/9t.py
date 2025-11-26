# !git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git
# %cd stylegan2-ada-pytorch

# !pip install requests pillow

# !wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl

import torch
import numpy as np
from PIL import Image
import dnnlib
import legacy

device = torch.device('cuda')
network_path = "./ffhq.pkl"

with open(network_path, 'rb') as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device)

z = torch.randn(1, G.z_dim, device=device)
img = G(z, None)

# Convert format
img = (img * 127.5 + 127.5).clamp(0,255).to(torch.uint8)
img = img[0].permute(1, 2, 0).cpu().numpy()

Image.fromarray(img).save("generated_face.png")
Image.fromarray(img)
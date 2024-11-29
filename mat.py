import cv2
import pyspng
import sys
import os
import re
import random
from typing import List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import dnnlib as dnnlib
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

import legacy as legacy
from datasets.mask_generator_512 import RandomMask
from networks.mat import Generator

import gen_old_image 

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]


def copy_params_and_buffers(src_module, dst_module, require_all=False):
    assert isinstance(src_module, torch.nn.Module)
    assert isinstance(dst_module, torch.nn.Module)
    src_tensors = {name: tensor for name, tensor in named_params_and_buffers(src_module)}
    for name, tensor in named_params_and_buffers(dst_module):
        assert (name in src_tensors) or (not require_all)
        if name in src_tensors:
            tensor.copy_(src_tensors[name].detach()).requires_grad_(tensor.requires_grad)


def params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.parameters()) + list(module.buffers())


def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())

def generate_images(
    network_pkl: str, # đường dẫn tới mô hình
    ipath: str,  # đường dẫn tới hình ảnh cần khôi phục
    imask: str,
    resolution = 512, # độ phân giải hình ảnh
    truncation_psi = 1,
    noise_mode = "const",
    outputpath = '../output.png',
):
    """
    Generate images using pretrained network pickle.
    """
    seed = 240  # pick up a random number
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


    print(f'Loading networks from: {network_pkl}')
    device = torch.device('cpu')
    with dnnlib.util.open_url(network_pkl) as f:
        G_saved = legacy.load_network_pkl(f)['G_ema'].to(device).eval().requires_grad_(False) # type: ignore
    net_res = 512 if resolution > 512 else resolution

    G = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=net_res, img_channels=3).to(device).eval().requires_grad_(False)
    copy_params_and_buffers(G_saved, G, require_all=True)

    # os.makedirs(outdir, exist_ok=True)

    # no Labels.
    label = torch.zeros([1, G.c_dim], device=device)

    def read_image(image_path):
        with open(image_path, 'rb') as f:
            if pyspng is not None and image_path.endswith('.png'):
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
            image = np.repeat(image, 3, axis=2)
        image = image.transpose(2, 0, 1) # HWC => CHW
        image = image[:3]
        return image

    def to_image(image, lo, hi):
        image = np.asarray(image, dtype=np.float32)
        image = (image - lo) * (255 / (hi - lo))
        image = np.rint(image).clip(0, 255).astype(np.uint8)
        image = np.transpose(image, (1, 2, 0))
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        return image

    if resolution != 512:
        noise_mode = 'random'
    with torch.no_grad():
        image = read_image(ipath)
        image = (torch.from_numpy(image).float().to(device) / 127.5 - 1).unsqueeze(0)

        mask = RandomMask(resolution)
        mask = torch.from_numpy(mask).float().to(device).unsqueeze(0)
        # mask = resize_tensor(mask, target_size=image.shape[2:])

        z = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)
        output = G(image, mask, z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        output = (output.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
        output = output[0].cpu().numpy()
        PIL.Image.fromarray(output, 'RGB').save(outputpath)

def main():
    # vintage_image_path, mask_path = gen_old_image.main('messi.jpg')
    vintage_image_path = 'messi_vintage_image.jpg'
    mask_path = 'messi_mask.jpg'
    generate_images(
        network_pkl = "models/Places_512_FullData.pkl",
        ipath = vintage_image_path,
        imask = mask_path,
        outputpath="output.png"
    )
    
main()
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from models.network_swinir import SwinIR
from collections import OrderedDict
import torch.nn.functional as F
import cv2
import os

def upscale(image_path, device='cpu'):
    model = SwinIR(
        upscale=4, in_chans=3, img_size=64, window_size=8,
        img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
        num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
        mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv'
    )

    param_key_g = 'params_ema'
    state_dict = torch.load('./models/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth')
    model.load_state_dict(state_dict[param_key_g] if param_key_g in state_dict.keys() else state_dict, strict=True)
    model.eval()
    model.to(device)

    # Load and pad the image
    img_lq = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))
    img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)

    with torch.no_grad():
        _, _, h_old, w_old = img_lq.size()
        h_pad = (h_old // 8 + 1) * 8 - h_old
        w_pad = (w_old // 8 + 1) * 8 - w_old
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
        output = model(img_lq)
        output = output[..., :h_old * 4, :w_old * 4]

    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    if output.ndim == 3:
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
    output = (output * 255.0).round().astype(np.uint8)
    
    cv2.imwrite(f'./static/output/upscaled_{os.path.basename(image_path)}', output)

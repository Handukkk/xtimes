#!/bin/sh

curl -L -O https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth -o experiments/pretrained_models
curl -L -O https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/004_grayDN_DFWB_s128w8_SwinIR-M_noise15.pth -o experiments/pretrained_models
curl -L -O https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/004_grayDN_DFWB_s128w8_SwinIR-M_noise25.pth -o experiments/pretrained_models
curl -L -O https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/004_grayDN_DFWB_s128w8_SwinIR-M_noise50.pth -o experiments/pretrained_models
curl -L -O https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth -o experiments/pretrained_models
curl -L -O https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth -o experiments/pretrained_models
curl -L -O https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth -o experiments/pretrained_models
curl -L -O https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/006_CAR_DFWB_s126w7_SwinIR-M_jpeg10.pth -o experiments/pretrained_models
curl -L -O https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/006_CAR_DFWB_s126w7_SwinIR-M_jpeg20.pth -o experiments/pretrained_models
curl -L -O https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/006_CAR_DFWB_s126w7_SwinIR-M_jpeg30.pth -o experiments/pretrained_models
curl -L -O https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/006_CAR_DFWB_s126w7_SwinIR-M_jpeg40.pth -o experiments/pretrained_models
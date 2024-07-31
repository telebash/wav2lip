#!/bin/bash

wget https://github.com/anothermartz/Easy-Wav2Lip/releases/download/Prerequesits/Wav2Lip_GAN.pth -O ./checkpoints/wav2lip_gan.pth
wget 'https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth' -O ./face_detection/detection/sfd/s3fd.pth

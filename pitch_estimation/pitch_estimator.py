# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 15:31:49 2022

@author: Admin
"""
from collections import OrderedDict
import numpy as np 
import torch.nn as nn
import torch
import torch.nn.functional as F
from signal_processing_full import STFT_module, iSTFT_module

class same_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(same_Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels = in_channels,
                              out_channels = out_channels,
                              kernel_size = kernel_size,
                              stride = stride)
        self.BN = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU(out_channels)    
        self.padding = nn.ConstantPad2d([0,0,kernel_size[0]//2,kernel_size[0]//2], value=0.)
    
    def forward(self, x):

        x = self.padding(x)
        x = self.act(self.BN(self.conv(x)))
        return x
        
class pitch_estimator(nn.Module):
    def __init__(self, n_pitch = 226, N_FFT = 1536, block_shift = 384, block_len=1536, N_feature = 160):
        super(pitch_estimator, self).__init__()
        self.n_pitch = n_pitch
        self.STFT_model = STFT_module(N_FFT, block_shift, block_len, center = True, mode = 'real_imag', device ='cpu')
        self.Conv = nn.Sequential(OrderedDict([
              ('conv1', same_Conv(1, 16, (3,3), (1,2))),
              ('conv2', same_Conv(16, 32, (3,3), (1,2))),
              ('conv3', same_Conv(32, 64, (3,3), (1,2))),
              ('conv4', same_Conv(64, 128, (3,3), (1,2))),
              ('conv5', same_Conv(128, 256, (3,3), (1,2))),
            ]))
        self.N_feature = N_feature
        self.GRU = nn.GRU(1024,64,batch_first = True,bidirectional=True)
        self.Dense = nn.Linear(128, n_pitch)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        """
        s: bs,T
        pitch_out: bs,T,N_picth
        """
        s = data['clean_wav']
        #print(s)
        s = s / (torch.max(s,dim=-1,keepdim=True)[0] + 1e-8)
        
        if self.STFT_model.window.device != s.device:
            self.STFT_model.window.to(s.device)

        spec = self.STFT_model(s)
        log_mag = torch.log(torch.norm(spec, dim = 1, keepdim=True)[:,:,:,:self.N_feature] + 1e-8)
        gru_in = self.Conv(log_mag)
        bs,C,T,F = gru_in.shape
        gru_in = torch.reshape(gru_in.permute([0,2,1,3]), [bs,-1,C*F])
        gru_out,_ = self.GRU(gru_in)
        
        pitch_out = self.sigmoid(self.Dense(gru_out))
        data['pred_pitch'] = pitch_out
        return data

if __name__ == '__main__':
    import librosa
    import matplotlib.pyplot as plt
    import soundfile as sf
    pe = pitch_estimator(n_pitch = 226, N_FFT = 1536, block_shift = 384, block_len=1536, N_feature = 160).to('cpu')
    pe.eval()
    checkpoint = torch.load('/home/nis/tong.lei/pjt6mic/FT-JNF/pitch_estimation/ckps/pitch.pth', map_location='cpu')
    new_state_dict = {}
    for k,v in checkpoint['state_dict'].items():
        new_state_dict[k[0:]] = v
    # print(new_state_dict)
    pe.load_state_dict(new_state_dict)

    pe.eval()
    s = sf.read('/home/nis/tong.lei/pjt6mic/FT-JNF/pitch_estimation/p232_065.wav', dtype = np.int16)[0]
    # s = s / np.max(abs(s))

    #pitch_label = get_pitch(s, center_freqs)

    with torch.no_grad():
        pred_pitch = pe({'clean_wav':torch.FloatTensor(s[None]/32767).cpu()})['pred_pitch'][0].T
    plt.figure(0)
    plt.imshow(pred_pitch.cpu())
    plt.show()
    # plt.savefig('./pitch.png')

import numpy as np 
import torch.nn as nn
import torch
from collections import OrderedDict
from espnet2.enh.encoder.stft_encoder import STFTEncoder

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
    def __init__(self, n_pitch = 226, N_FFT = 512, block_shift = 256, block_len=512, N_feature = 160):
        super(pitch_estimator, self).__init__()
        self.n_pitch = n_pitch
        # self.STFT_model = STFT_module(N_FFT, block_shift, block_len, center = True, mode = 'real_imag', device ='cpu')
        self.STFT_model = STFTEncoder(N_FFT, win_length=N_FFT, hop_length=block_shift, window="hann")
        self.Conv = nn.Sequential(OrderedDict([
              ('conv1', same_Conv(1, 16, (3,3), (1,2))),
              ('conv2', same_Conv(16, 32, (3,3), (1,2))),
              ('conv3', same_Conv(32, 64, (3,3), (1,2))),
              ('conv4', same_Conv(64, 128, (3,3), (1,2))),
              ('conv5', same_Conv(128, 256, (3,3), (1,2))),
            ]))
        self.N_feature = N_feature
        self.GRU = nn.LSTM(1024,256,batch_first = True,bidirectional=True)
        self.GRU1 = nn.LSTM(512,128,batch_first = True,bidirectional=True)
        self.GRU2 = nn.LSTM(256,64,batch_first = True,bidirectional=True)
        self.Dense = nn.Linear(128, n_pitch)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        """
        s: bs,T
        pitch_out: bs,T,N_picth
        """
        s = data
        #print(s)
        s = s / (torch.max(s,dim=-1,keepdim=True)[0] + 1e-8)

        spec = self.STFT_model(s[..., None], None)[0]
        spec = spec.transpose(1, 2)  # [B, M, T, F]
        spec = torch.cat((spec.real, spec.imag), dim=1)  # [B, 2*M, T, F]

        log_mag = torch.log(torch.norm(spec, dim = 1, keepdim=True)[:,:,:,:self.N_feature] + 1e-8)

        gru_in = self.Conv(log_mag)
        bs,C,T,F = gru_in.shape

        gru_in = torch.reshape(gru_in.permute([0,2,1,3]), [bs,-1,C*F])

        gru_out,_ = self.GRU(gru_in)
        gru_out,_ = self.GRU1(gru_out)
        gru_out,_ = self.GRU2(gru_out)
        
        
        pitch_out = self.sigmoid(self.Dense(gru_out))
        # data['pred_pitch'] = pitch_out
        return pitch_out

if __name__ == '__main__':
    model = pitch_estimator()
    x = torch.randn([3,16000*8])
    y = model(x)
    print(y.shape)
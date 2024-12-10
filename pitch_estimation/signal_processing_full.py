# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 16:27:34 2022

@author: lexiaohuai
"""
import torch
from torch import nn
import numpy as np

class subbands():

    def __init__(self, n_bands, fl = 20, fh = 8000, fs = 16000, n_fft = 512):
        self.n_bands = n_bands
        self.fl = fl
        self.fh = fh
        self.n_fft = 512
        self.fs = fs 
        
    @staticmethod
    def get_triangular_filter(band_f, n_filt, n_fft):

        fbank = np.zeros([n_filt, n_fft // 2 + 1])
        
        for index_band in range(0, n_filt):
              left = band_f[index_band]
              center = band_f[index_band + 1]
              right = band_f[index_band + 2]
              for indexi in range(int(left), int(center)):
                  fbank[index_band, indexi] = (indexi - left) / ( center - left)
              for indexi in range(int(center), int(right)):
                  fbank[index_band, indexi] = (right - indexi) / ( right -center)
        return fbank
    
    
class mel(subbands):
    '''
    generate mel sub-bands
    '''        
    @staticmethod
    def hz2mel(hz):
        return 2595 * np.log10(1 + hz / 700.)
    
    @staticmethod
    def mel2hz(mel):
        return 700 * (10 ** (mel / 2595.0) - 1)
    
    def get_mel_matrix(self,):
        
        low_mel = self.hz2mel(self.fl)
        high_mel = self.hz2mel(self.fh)
        mel_points = np.linspace(low_mel, high_mel, self.n_bands + 2)
        
        band_f = np.round((self.n_fft) * self.mel2hz(mel_points) / self.fs)
        
        mel_fbank = self.get_triangular_filter(band_f, self.n_bands, self.n_fft)
        return mel_fbank
    
    def get_mel_inv_matrix(self,):
        '''
        transform the mel spectrum to the linear spectrum by linear interplation
        '''
        
    def inv_mel(self, mel_feature, mel_mat):
        '''
        transform the mel spectrum to the linear spectrum by linear interplation
        '''
        '''
        feature = np.zeros([self.n_fft //2+1, mel_feature.shape[1]])
        for i in range(mel_feature.shape[1]):
            m = mel_feature[:,i:i+1] * mel_mat
            feature[:,i] = np.sum(m,0)
        return feature
        '''
        return (mel_feature.T @ mel_mat).T
        
class erb(subbands):
    '''
    generate erb sub-bands
    '''
    @staticmethod
    def hz2erb(hz):
        A = 1000 * np.log(10.) / ( 24.7 * 4.37)
        erb = A * np.log10(1 + hz * 0.00437)
        return erb        
    @staticmethod
    def erb2hz(erb):
        A = 1000 * np.log(10.) / ( 24.7 * 4.37)
        hz = (10 ** (erb / A) - 1) / 0.00437
        return hz
    
    def get_erb_matrix(self,):
        
        low_erb = self.hz2erb(self.fl)
        high_erb = self.hz2erb(self.fh)
        erb_points = np.linspace(low_erb, high_erb, self.n_bands + 2)
        
        band_f = np.round((self.n_fft) * self.erb2hz(erb_points) / self.fs)
        
        erb_fbank = self.get_triangular_filter(band_f, self.n_bands, self.n_fft)
        return erb_fbank
    
    def get_erb_inv_matrix(self,):
        '''
        transform the erb spectrum to the linear spectrum by linear interplation
        '''
        
    def inv_erb(self,):
        '''
        transform the erb spectrum to the linear spectrum by linear interplation
        '''
        pass
    
class bark(subbands):
    '''
    generate bark sub-bands
    '''
    @staticmethod
    def hz2bark(hz):
        bark = 26.81 * hz / (1960 + hz) - 0.53
        if bark < 2:
            bark = bark + 0.15 * (2 - bark)
        if bark > 20.1:
            bark = bark + 0.22 * (bark - 20.1)
        return bark
    
    @staticmethod
    def bark2hz(bark):
        if bark < 2:
            bark = (bark - 0.3) / 0.85
        if bark > 20.1:
            bark = (bark + 4.422) / 1.22
        hz = 1960 * ( (bark + 0.53) / (26.28 - bark))
        return hz
    
    def get_bark_matrix(self,):

        low_bark = self.hz2bark(self.fl)
        high_bark = self.hz2bark(self.fh)

        bark_points = np.linspace(low_bark, high_bark, self.n_bands + 2)
        
        band_f = np.zeros(len(bark_points))
        for index in range(0, len(bark_points)):
            band_f[index] = np.round((self.n_fft) * self.bark2hz(bark_points[index]) / self.fs)
        
        bark_fbank = self.get_triangular_filter(band_f, self.n_bands, self.n_fft)
        return bark_fbank
    
    def get_bark_inv_matrix(self,):
        '''
        transform the bark spectrum to the linear spectrum by linear interplation
        '''
        
    def inv_bark(self,):
        '''
        transform the bark spectrum to the linear spectrum by linear interplation
        '''
        pass
    


'''
STFT module for torch>=1.7
'''
class STFT_module(nn.Module):
    def __init__(self, n_fft, hop_length, win_length, center = True, normalized = False, window = torch.hann_window, mode = 'real_imag', device = 'cuda'):
         super(STFT_module, self).__init__()
         self.mode = mode
         self.n_fft = n_fft
         self.hop_length = hop_length
         self.win_length = win_length
         self.center = center
         self.normalized = normalized
         norm_coef = self.win_length //2 //self.hop_length
         if not window:
             #sinwin
             self.window = torch.sqrt(window(self.win_length)/norm_coef+1e-8).to(device)
         else:
             self.window = window(self.win_length).to(device)

    def forward(self, x):
         '''
         return: batchsize, 2, Time, Freq
         '''
         self.window = self.window.to(x.device)
         spec_complex = torch.stft(x, n_fft=self.n_fft, 
                                   hop_length=self.hop_length, 
                                   win_length=self.win_length,
                                   center=self.center,
                                   window=self.window,
                                   normalized=self.normalized,
                                   return_complex=False)
         if self.mode == 'real_imag':
             #return torch.permute(spec_complex,[0, 3, 2, 1])
             return spec_complex.permute([0, 3, 2, 1]).contiguous()
         
         elif self.mode == 'mag_pha':
             
             #spec_complex = torch.permute(spec_complex,[0, 3, 2, 1])
             spec_complex = spec_complex.permute([0, 3, 2, 1]).contiguous()
             mag = torch.sqrt(spec_complex[:, 0, :, :]**2 + spec_complex[:, 1, :, :]**2)
             angle = torch.atan2(spec_complex[:, 1, :, :],spec_complex[:, 0, :, :])
             return torch.stack([mag,angle],1)


'''
iSTFT module for torch >= 1.8
'''
class iSTFT_module(nn.Module):

    def __init__(self, n_fft, hop_length, win_length, length, center = False, window = torch.hann_window, mode = 'real_imag',device = 'cuda'):
        super(iSTFT_module, self).__init__()
        self.mode = mode
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.length = length
        self.center = center
        norm_coef = self.win_length //2 //self.hop_length
        if center:
            self.padding_num = int((self.win_length / 2 ) // (self.hop_length) * self.hop_length)
        else:
            self.padding_num = 0
            
        if not window:
            self.window = torch.sqrt(torch.hann_window(self.win_length)/norm_coef+1e-8).to(device)
        else:
            self.window = window(self.win_length).to(device)
            
    def forward(self, x):
        '''
        x: batchsize, 2, Time, Freq
        '''
        self.window = self.window.to(x.device)
        length = self.win_length + self.hop_length * (x.shape[-2] - 1)
        x = x.permute([0, 3, 2, 1]).contiguous()
        
        s = torch.istft(x, n_fft=self.n_fft, 
                       hop_length=self.hop_length, 
                       win_length=self.win_length,
                       center=self.center,
                       window=self.window,
                       normalized=False)
        return s
    
        
class Signal_Pro():
    def __init__(self, config, length, device = 'cuda:1'):
        
        self.fs = config['stft']['fs']
        self.block_len = config['stft']['block_len']
        self.block_shift = config['stft']['block_shift']
        self.window = config['stft']['window']
        self.N_FFT = config['stft']['N_FFT']
        self.win = None
        if length is None:
            L = None
            length_points = None
        else:
            L = (16000 * length - self.block_len) // self.block_shift + 1
            length_points = self.block_len + self.block_shift * (L-1)
        #print(L,length_points)
        self.STFT_module = STFT_module(self.N_FFT, self.block_shift, self.block_len, center = True, mode = 'real_imag', device =device)
        self.iSTFT_module = iSTFT_module(self.N_FFT, self.block_shift, self.block_len, length = length_points, center = True, mode = 'real_imag', device =device)
     
    def mk_mask_complex(self, noisy, mask):
        '''
        complex ratio mask
        '''
        enh_real = noisy[:,0:1,:,:] * mask[:,0:1,:,:] - noisy[:,1:2,:,:] * mask[:,1:2,:,:]
        enh_imag = noisy[:,0:1,:,:] * mask[:,1:2,:,:] + noisy[:,1:2,:,:] * mask[:,0:1,:,:]
        
        return torch.cat([enh_real,enh_imag],dim=1)
    
    def mk_mask_mag(self, noisy, mag_mask):
        '''
        magnitude mask
        '''
        return noisy * mag_mask
    
    def mk_mask_pha(self, x):
        '''
        phase mask
        '''
        [enh_mag,pha_cos,pha_sin] = x
        
        enh_real = enh_mag * pha_cos - enh_mag * pha_sin
        enh_imag = enh_mag * pha_sin + enh_mag * pha_cos
        
        return [enh_real,enh_imag]
    
    def mk_mask_mag_pha(self, x):
        
        [noisy_real,noisy_imag,mag_mask,pha_cos,pha_sin] = x
        noisy_real = noisy_real[:,:,:,0]
        noisy_imag = noisy_imag[:,:,:,0]
        
        enh_mag_real = noisy_real * mag_mask
        enh_mag_imag = noisy_imag * mag_mask
        
        enh_real = enh_mag_real * pha_cos - enh_mag_imag * pha_sin
        enh_imag = enh_mag_real * pha_sin + enh_mag_imag * pha_cos
        
        return [enh_real,enh_imag]

if __name__ == '__main__':
    import soundfile as sf
    import librosa

    s = sf.read('D:/codes/test_audio/clean/444C020a.wav')[0]
    n = sf.read('D:/codes/test_audio/mix/444C020a_mix.wav')[0]
    w = np.sum(s*n) / np.sum(s*s)
    s = w * s
    noise = n - s
    
    spec_s = librosa.stft(s, 512, 256)
    spec_noisy = librosa.stft(n, 512, 256)
    spec_noise = librosa.stft(noise, 512, 256)
    amp_s = abs(spec_s)
    amp_noisy = abs(spec_noisy)
    amp_noise = abs(spec_noise)
    
    mel_b = mel(64)
    mel_mat = mel_b.get_mel_matrix()
    
    mel_s = mel_mat @ amp_s
    mel_noisy = mel_mat @ amp_noisy
    mel_noise = mel_mat @ amp_noise
    
    mel_mask = mel_s / (mel_noisy+1e-8)
    L = mel_mask.shape[1]
    mask = np.zeros_like(amp_s)
    for i in range(L):
        m = mel_s[:,i:i+1] * mel_mat
        mask[:,i] = np.sum(m,0)
    
    
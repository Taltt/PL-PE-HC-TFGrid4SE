from random import random
import soundfile as sf
import librosa
import torch
import os
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd


TRAIN_CLEAN_CSV = '/data/sdd2/tong.lei/Data/train_clean2_data.csv'
TRAIN_NOISE_CSV = '/data/sdd2/tong.lei/Data/train_noise2_data.csv'
VALID_CLEAN_CSV = '/data/sdd2/tong.lei/Data/valid_clean2_data.csv'
VALID_NOISE_CSV = '/data/sdd2/tong.lei/Data/valid_noise2_data.csv'
RIR_DIR = '/data/sdd2/tong.lei/Data/rir_SLR26/simulated_rirs_16k/'
#ground_n = sf.read('./ground_noise.wav')[0][:3*16000]

T = int(500 * 16000 / 1000) 
t = np.arange(16000)
h = np.exp(-6 * np.log(10) * t / T)

# FIR_LOW = []
# for cut_freq in range(16, 40):
#     fir = signal.firwin(128, cut_freq /48.0)
#     FIR_LOW.append(fir)




def add_pyreverb(clean_speech, rir):
    # max_index = np.argmax(np.abs(rir))
    # rir = rir[max_index:]
    reverb_speech = signal.fftconvolve(clean_speech, rir, mode="full")
    
    # make reverb_speech same length as clean_speech
    reverb_speech = reverb_speech[: clean_speech.shape[0]]

    return reverb_speech

def mk_mixture(s1,s2,snr,eps = 1e-8):

    s1 = s1 / (np.max(np.abs(s1)) + eps) 
    norm_sig1 = s1
    norm_sig2 = s2 * np.math.sqrt(np.sum(s1 ** 2) + eps) / np.math.sqrt(np.sum(s2 ** 2) + eps)

    alpha_4 = 10**(-(snr*1.5+20)/20)
    
    freq_num = np.random.randint(0,4)
    sins = np.zeros(len(s1))
    if freq_num > 1:
        freq = np.random.choice(range(50,8000),freq_num)
        for f in freq:
            s_sin = np.sin(2*np.pi*f*np.arange(len(s1))/16000)
            sins = sins + (0.5*np.random.rand() + 0.5) * s_sin * np.math.sqrt(np.sum(s1 ** 2) + eps) / np.math.sqrt(np.sum(s_sin ** 2) + eps)

    mix_4 = norm_sig1 + alpha_4 * (norm_sig2 + sins)


    return mix_4

class Dataset(torch.utils.data.Dataset):
    def __init__(self, fs=16000, length_in_seconds=8, random_start_point=False, train=True):
        self.train_clean_list = pd.read_csv(TRAIN_CLEAN_CSV)['file_dir'].to_list()[:]#只用了前10000
        self.train_noise_list = pd.read_csv(TRAIN_NOISE_CSV)['file_dir'].to_list()[:]
        self.valid_clean_list = pd.read_csv(VALID_CLEAN_CSV)['file_dir'].to_list()[:1000]
        self.valid_noise_list = pd.read_csv(VALID_NOISE_CSV)['file_dir'].to_list()[:1000]
        self.train_snr_list = pd.read_csv(TRAIN_CLEAN_CSV)['snr'].to_list()
        self.valid_snr_list = pd.read_csv(VALID_CLEAN_CSV)['snr'].to_list()
        self.L = int(length_in_seconds * fs)
        self.random_start_point = random_start_point
        self.fs = fs
        self.length_in_seconds = length_in_seconds
        self.train = train
        self.rir_list = librosa.util.find_files(RIR_DIR,ext = 'wav')
        self.center_freqs = np.load('/home/nis/tong.lei/pjt6mic/FT-JNF/pitch_estimation/ckps/linear_center_freqs.npy')
        self.ground = False
        #print('%s audios for training, %s for validation' %(len(self.train_clean_list), len(self.valid_clean_list)))
        
    def pitch_smooth(self, pitch):
        # pitch: T
        pitch_label = np.zeros([len(pitch),len(self.center_freqs) + 1])
        index = np.arange(len(self.center_freqs) + 1)
        for i in range(len(pitch)):
            pitch_label[i,:] = np.exp(- (index - pitch[i]) ** 2 / (2 * 5**2))
        return pitch_label.astype(np.float32)

    def get_pitch(self, s, frame_length=1024,hop_length=256):
        s = s / max(np.abs(s)+1e-8)
        f0 = librosa.pitch.pyin(s,62.5,500,16000,frame_length=frame_length,hop_length=hop_length,center=True)
        pitch = f0[0]
        pitch[np.isnan(pitch)] = -225
        freq_samples = self.center_freqs
        pitch = self.hz_to_points(freq_samples, pitch)
        return pitch.astype('int64')

    @staticmethod
    def hz_to_points(freq_samples, pitch):
        for i,f in enumerate(pitch):
            if f > 0 :
                pitch[i] = (np.abs(freq_samples-f)).argmin()
        pitch[pitch<0] = 225
        return pitch
    
    def __getitem__(self, idx):
        if self.train:
            clean_list = self.train_clean_list
            noise_list = self.train_noise_list
            snr_list = self.train_snr_list
        else:
            clean_list = self.valid_clean_list
            noise_list = self.valid_noise_list 
            snr_list = self.valid_snr_list           
        # reverb_rate = 0 #np.random.rand()
        # clip_rate = 1.0 #np.random.rand()


        if self.random_start_point:
            Begin_S = int(np.random.uniform(0,30 - self.length_in_seconds)) * self.fs
            Begin_N = int(np.random.uniform(0,10 - self.length_in_seconds)) * self.fs
            clean, sr_s = sf.read(clean_list[idx], dtype='float32',start= Begin_S,stop = Begin_S + self.L)
            noise, sr_n = sf.read(noise_list[idx % len(noise_list)], dtype='float32',start= Begin_N,stop = Begin_N + self.L)

        else:
            clean, sr_s = sf.read(clean_list[idx], dtype='float32',start= 0,stop = self.L) 
            noise, sr_n = sf.read(noise_list[idx % len(noise_list)], dtype='float32',start= 0,stop = self.L)

            
        # if reverb_rate < 0.1: # 妯℃嫙娣峰搷淇″彿
        rir_idx = np.random.randint(0,len(self.rir_list) - 1)
        rir_f = self.rir_list[rir_idx]
        rir_s = sf.read(rir_f,dtype = 'float32')[0]
        if len(rir_s.shape)>1:
            rir_s = rir_s[:,0]
        max_index = np.argmax(np.abs(rir_s))
        rir_s = rir_s[max_index:]
        reverb = add_pyreverb(clean, rir_s)
            
        mix_4 = mk_mixture(reverb,noise,snr_list[idx],eps = 1e-8)

        pitch = self.get_pitch(clean)
        pitch_smth = self.pitch_smooth(pitch)

        return mix_4.astype(np.float32), pitch_smth


    def __len__(self):
        if self.train:
            return len(self.train_clean_list)
        else:
            return len(self.valid_clean_list)

def collate_fn(batch):

    noisy, clean = zip(*batch)
    noisy = np.asarray(noisy)
    clean = np.asarray(clean)
    return noisy, clean 

if __name__=='__main__':
    dataset = Dataset(length_in_seconds=10, random_start_point=True, train=False)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    for i, data in enumerate(train_loader):
        mix, pitch_smth = data
        # sf.write('./noisy.wav', noisy_s[0,:].detach().cpu().numpy(), 16000)
        # sf.write('./mix_1.wav', mix_1[0,:].detach().cpu().numpy(), 16000)
        # sf.write('./mix_2.wav', mix_2[0,:].detach().cpu().numpy(), 16000)
        # sf.write('./mix_3.wav', mix_3[0,:].detach().cpu().numpy(), 16000)
        # sf.write('./mix_4.wav', mix_4[0,:].detach().cpu().numpy(), 16000)
        # sf.write('./s_early.wav', s_early[0,:].detach().cpu().numpy(), 16000)
        break

import torch
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import soundfile as sf
from data_generator import Dataset, collate_fn
from pitch_estimator_model import pitch_estimator
# device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
import torch.nn.functional as F
from signal_processing_full import iSTFT_module
torch.set_default_tensor_type(torch.FloatTensor)
device_ids = [1,2]
import librosa
import pesq
from collections import OrderedDict
import os

dataset = Dataset(length_in_seconds=10, random_start_point=True, train=False)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
for i, data in enumerate(train_loader):
    mix, pitch_smth = data
    break

model = pitch_estimator().to(device) # 定义模型
# model_cpu = MTFAA_Net().to('cpu')

''' train from checkpoints'''
checkpoint_path = '/data/ssd1/tong.lei/exp_FTJNF/Pitch_estimator/chkpt/epoch_99_trainloss_0.060945485_validloss_0.06246999.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)
new_state_dict = OrderedDict()
for k, v in checkpoint['state_dict'].items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)   
model.eval()
with torch.no_grad():
    mix = mix.to(device)
    pitch_est = model(mix) 
scale_mix = mix / torch.max(torch.abs(mix), 1)[0][:,None]

from comb_filter import comb_filter
cf = comb_filter(mode = 'linear', center=True).to(device)
cf.window = cf.window.to(device)
pitch_input = torch.argmax(pitch_est, dim=-1)
filtered_mix = cf(scale_mix, pitch_input)[:,:,:,0]
filtered_spec = torch.fft.rfft(filtered_mix,dim=2)
filtered_spec = torch.stack([filtered_spec.real,filtered_spec.imag], dim=1)
filter_s = torch.istft(filtered_spec.permute(0,3,2,1), 512, 256, 512, torch.hann_window(512).to(device))

sf.write('./mix.wav', mix[0,:].detach().numpy(), 16000)
sf.write('./filter_s.wav', filter_s[0,:].detach().numpy(), 16000)
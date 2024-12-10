# from typing import OrderedDict
# from unicodedata import name
import torch
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import soundfile as sf
from data_generator import Dataset, collate_fn
from pitch_estimator_model import pitch_estimator
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
import torch.nn.functional as F
torch.set_default_tensor_type(torch.FloatTensor)
device_ids = [1,0]
# import librosa
# import pesq
from collections import OrderedDict
import os

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

def train(end_epoch = 100):


    '''model'''
    model = pitch_estimator() # 定义模型
    # model_cpu = MTFAA_Net().to('cpu')

    ''' train from checkpoints'''
    checkpoint_path = '/data/ssd1/tong.lei/exp_FTJNF/Pitch_estimator/chkpt/epoch_99_trainloss_0.060945485_validloss_0.06246999.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    '''multi gpu'''
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = model.to(device)
    


    '''optimizer & lr_scheduler'''
    optimizer = NoamOpt(model_size=32, factor=0.2, warmup=600,
                    optimizer=torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))  
    

    '''load train data'''
    dataset = Dataset(length_in_seconds=8, random_start_point=True, train=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=96, shuffle=True, drop_last=True, num_workers=16)

    '''start train'''
    for epoch in range(end_epoch):
        train_loss = []
        model.train()
        dataset.train = True
        dataset.random_start_point = True
        idx = 0

        '''train'''
        print('epoch %s--training' %(epoch))
        for i, data in enumerate(tqdm(data_loader)):
            noisy, pitch_label = data
            noisy = noisy.to(device)
            pitch_label = pitch_label.to(device)
            optimizer.optimizer.zero_grad() #使用之前先清零 warm up
            pitch_est = model(noisy)       
            loss = F.binary_cross_entropy(pitch_est, pitch_label)
            loss.backward() # loss反传，计算模型中各tensor的梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
            optimizer.step()
            train_loss.append(loss.cpu().detach().numpy())
            idx += 1
        train_loss = np.mean(train_loss) # 对各个mini batch的loss求平均

        '''eval'''
        valid_loss = []
        model.eval()  # 注意model的模式从train()变成了eval()
        print('epoch %s--validating' %(epoch))
        dataset.train = False
        dataset.random_start_point = False
        with torch.no_grad():
            for i, data in enumerate(tqdm(data_loader)):
                noisy, pitch_label = data
                noisy = noisy.to(device)
                pitch_label = pitch_label.to(device)
                pitch_est = model(noisy)   
                loss = F.binary_cross_entropy(pitch_est, pitch_label)
                valid_loss.append(loss.cpu().detach().numpy())
            valid_loss = np.mean(valid_loss)
        print('train loss: %s, valid loss %s' %(train_loss, valid_loss))

        torch.save(
            {'epoch': epoch,
                'state_dict': model.state_dict(),
                # 'optimizer': optimizer_dpcrn.state_dict()}, # optimizer_dpcrn.optimizer.state_dict() 如果是warmup用这句
                'optimizer': optimizer.optimizer.state_dict()},
            '/data/ssd1/tong.lei/exp_FTJNF/Pitch_estimator/chkpt/epoch_%s_trainloss_%s_validloss_%s.pth' %(str(epoch), str(train_loss), str(valid_loss)))

if __name__ == '__main__':
    train(end_epoch=100)



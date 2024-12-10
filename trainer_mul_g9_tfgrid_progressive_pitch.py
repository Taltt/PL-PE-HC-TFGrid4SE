'''
multi gpu version
'''
import os
import torch
import toml
from datetime import datetime
from tqdm import tqdm
from glob import glob
import soundfile as sf
import numpy as np
import einops
from torch.utils.tensorboard import SummaryWriter
from pystoi import stoi


class Trainer:
    def __init__(self, config, model, optimizer, loss_func,
                 train_dataloader, validation_dataloader,train_sampler, args):
        self.model_pret = model[0]
        self.model_pitch = model[1]
        self.model_comb = model[2]
        self.model = model[-1]
        self.model_pret.eval()
        self.model_pitch.eval()
        self.model_comb.eval()
        
        self.optimizer = optimizer
        #self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.8, patience=5,verbose=True)
        self.loss_func = loss_func

        self.train_dataset = train_dataloader
        self.validation_dataloader = validation_dataloader

        self.train_sampler = train_sampler
        self.rank = args.rank
        self.device = args.device
        #self.WINDOW = torch.sqrt(torch.hann_window(512) + 1e-8).to(self.device)

        # training config
        self.trainer_config = config['trainer']
        self.epochs = self.trainer_config['epochs']
        self.save_checkpoint_interval = self.trainer_config['save_checkpoint_interval']
        self.clip_grad_norm_value = self.trainer_config['clip_grad_norm_value']
        self.resume = self.trainer_config['resume']

        if not self.resume:
            self.exp_path = self.trainer_config['exp_path'] + '_' + datetime.now().strftime("%Y-%m-%d-%Hh%Mm")
        
        else:
            self.exp_path = self.trainer_config['exp_path'] + '_' + self.trainer_config['resume_datetime']

        self.log_path = os.path.join(self.exp_path, 'logs')
        self.checkpoint_path = os.path.join(self.exp_path, 'checkpoints')
        self.sample_path = os.path.join(self.exp_path, 'val_samples')

        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.sample_path, exist_ok=True)
        #os.makedirs(self.sample_path + '/spk1', exist_ok=True)
        #os.makedirs(self.sample_path + '/spk2', exist_ok=True)
        #os.makedirs(self.sample_path + '/spk3', exist_ok=True)

        # save the config
        if self.rank == 1:
            with open(
                os.path.join(
                    self.exp_path, 'config_g8.toml'.format(datetime.now().strftime("%Y-%m-%d-%Hh%Mm"))), 'w') as f:

                toml.dump(config, f)

            self.writer = SummaryWriter(self.log_path)

        self.start_epoch = 1
        self.best_score = 0

        if self.resume:
            self._resume_checkpoint()

        self.sr = config['listener']['listener_sr']

        self.loss_func = self.loss_func.to(self.device)


    def _set_train_mode(self):
        self.model.train()
        #print(self.model.state_dict()['module.fc.bias'])

    def _set_eval_mode(self):
        self.model.eval()
        #print(self.model.state_dict()['module.fc.bias'])

    def _save_checkpoint(self, epoch, score):
        state_dict = {'epoch': epoch,
                      'optimizer': self.optimizer.optimizer.state_dict(),
                      'model': self.model.state_dict()}

        torch.save(state_dict, os.path.join(self.checkpoint_path, f'model_{str(epoch).zfill(4)}.tar'))

        if score > self.best_score:
            self.state_dict_best = state_dict.copy()
            self.best_score = score

    def _resume_checkpoint(self):
        latest_checkpoints = sorted(glob(os.path.join(self.checkpoint_path, 'model_*.tar')))[-1]

        map_location = self.device
        checkpoint = torch.load(latest_checkpoints, map_location=map_location)

        self.start_epoch = checkpoint['epoch'] + 1
        self.optimizer.optimizer.load_state_dict(checkpoint['optimizer'])
        
        self.model.load_state_dict(checkpoint['model'])#module.

    def _train_epoch(self, epoch):

        total_loss = 0
        self.train_dataset.dataset.sample()
        self.train_dataloader = tqdm(self.train_dataset, ncols=70)#tqdm(self.train_dataloader, ncols=75)
        '''
        for param_group in self.optimizer.optimizer.param_groups:
            lr = param_group['lr']
            print(lr)
        '''
        for step, data in enumerate(self.train_dataloader, 1):
            mixture,_,_,_,_,target = data
            mixture = mixture.to(torch.float32).to(self.device)                            # [B,N,M]
            target = target.to(torch.float32).to(self.device)                              # [B, N]
            with torch.no_grad():
                esti_tagt = self.model_pret(mixture)                   # [B,N,M]
                batch = esti_tagt[1]
                pitch_est = self.model_pitch(esti_tagt[0][-1])
                scale_mix0 = mixture / torch.max(torch.abs(mixture), 1)[0][:,None]
                scale_mix1 = esti_tagt[0][0] / torch.max(torch.abs(esti_tagt[0][0]), 1)[0][:,None]
                scale_mix2 = esti_tagt[0][1] / torch.max(torch.abs(esti_tagt[0][1]), 1)[0][:,None]
                scale_mix3 = esti_tagt[0][2] / torch.max(torch.abs(esti_tagt[0][2]), 1)[0][:,None]
                
                pitch_input = torch.argmax(pitch_est, dim=-1)

                filtered_mix0 = self.model_comb(scale_mix0, pitch_input)[:,:,:,0]
                filtered_spec0 = torch.fft.rfft(filtered_mix0,dim=2)
                filtered_spec0 = torch.stack([filtered_spec0.real,filtered_spec0.imag], dim=1) # [B, 2, T, F]

                filtered_mix1 = self.model_comb(scale_mix1, pitch_input)[:,:,:,0]
                filtered_spec1 = torch.fft.rfft(filtered_mix1,dim=2)
                filtered_spec1 = torch.stack([filtered_spec1.real,filtered_spec1.imag], dim=1)
    
                filtered_mix2 = self.model_comb(scale_mix2, pitch_input)[:,:,:,0]
                filtered_spec2 = torch.fft.rfft(filtered_mix2,dim=2)
                filtered_spec2 = torch.stack([filtered_spec2.real,filtered_spec2.imag], dim=1)
          
                filtered_mix3 = self.model_comb(scale_mix3, pitch_input)[:,:,:,0]
                filtered_spec3 = torch.fft.rfft(filtered_mix3,dim=2)
                filtered_spec3 = torch.stack([filtered_spec3.real,filtered_spec3.imag], dim=1)
                
                inp = [mixture, filtered_spec0, filtered_spec1, filtered_spec2, filtered_spec3, batch]
            
            esti_tagt = self.model(inp)
            #print(target.shape,esti_tagt[0][0].shape)
            loss = self.loss_func(esti_tagt[0][0],target)#+ self.loss_func(esti_nois, noise)
            loss = torch.mean(loss)#bs!=1时
            total_loss += loss.item()

            self.train_dataloader.desc = 'train[{}/{}]'.format(
                epoch, self.epochs + self.start_epoch-1)

            self.train_dataloader.postfix = 'ls{:.2f}'.format(total_loss / step)

            self.optimizer.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm_value)
            self.optimizer.step()
            #for name, parms in self.model.named_parameters():
            #    if 'fc.bias' in name:print(print('-->grad_value:',parms.grad))

         # 等待所有进程计算完毕
        if self.device != torch.device("cpu"):
            torch.cuda.synchronize(self.device)

        for param_group in self.optimizer.optimizer.param_groups:
            lr = param_group['lr']

        if self.rank == 1:
            self.writer.add_scalars('train_loss', {'train_loss': total_loss / step,'lr':lr}, epoch)


    @torch.no_grad()
    def _validation_epoch(self, epoch):
        total_loss = 0
        total_stoi_score = 0

        self.validation_dataloader = tqdm(self.validation_dataloader, ncols=70)
        for step, data in enumerate(self.validation_dataloader, 1):
            mix,_,_,_,_,tgt,name = data
            mix = mix.to(torch.float32).to(self.device)  
            tgt = tgt.to(torch.float32).to(self.device)
            esti_tagt = self.model_pret(mix)                   # [B,N,M]
            batch = esti_tagt[1]
            pitch_est = self.model_pitch(esti_tagt[0][-1])
            scale_mix0 = mix / torch.max(torch.abs(mix), 1)[0][:,None]
            scale_mix1 = esti_tagt[0][0] / torch.max(torch.abs(esti_tagt[0][0]), 1)[0][:,None]
            scale_mix2 = esti_tagt[0][1] / torch.max(torch.abs(esti_tagt[0][1]), 1)[0][:,None]
            scale_mix3 = esti_tagt[0][2] / torch.max(torch.abs(esti_tagt[0][2]), 1)[0][:,None]
            
            pitch_input = torch.argmax(pitch_est, dim=-1)

            filtered_mix0 = self.model_comb(scale_mix0, pitch_input)[:,:,:,0]
            filtered_spec0 = torch.fft.rfft(filtered_mix0,dim=2)
            filtered_spec0 = torch.stack([filtered_spec0.real,filtered_spec0.imag], dim=1) # [B, 2, T, F]

            filtered_mix1 = self.model_comb(scale_mix1, pitch_input)[:,:,:,0]
            filtered_spec1 = torch.fft.rfft(filtered_mix1,dim=2)
            filtered_spec1 = torch.stack([filtered_spec1.real,filtered_spec1.imag], dim=1)

            filtered_mix2 = self.model_comb(scale_mix2, pitch_input)[:,:,:,0]
            filtered_spec2 = torch.fft.rfft(filtered_mix2,dim=2)
            filtered_spec2 = torch.stack([filtered_spec2.real,filtered_spec2.imag], dim=1)
        
            filtered_mix3 = self.model_comb(scale_mix3, pitch_input)[:,:,:,0]
            filtered_spec3 = torch.fft.rfft(filtered_mix3,dim=2)
            filtered_spec3 = torch.stack([filtered_spec3.real,filtered_spec3.imag], dim=1)
            
            inp = [mix, filtered_spec0, filtered_spec1, filtered_spec2, filtered_spec3, batch]
            # FT-JNF
            esti_tagt = self.model(inp)                   # [B, F, T, 2]

            #print(tgt.shape,esti_tagt[0][0].shape)
            loss = self.loss_func(esti_tagt[0][0],tgt)# + self.loss_func(esti_nois, noise)
            total_loss += loss.item()

            enhanced = esti_tagt[0][0].squeeze().cpu().numpy()
            clean = tgt.squeeze().cpu().numpy()

            enh_len = enhanced.shape[-1]
            ###
            stoi_score = stoi(enhanced, clean[0:enh_len], 16000, extended=True)
            total_stoi_score += stoi_score

            if step<3:
                sf.write(os.path.join(self.sample_path,
                                    '{}_enhanced_epoch{}_estoi={:.3f}.wav'.format(name[0],epoch, stoi_score)),
                                    enhanced, 16000)
                sf.write(os.path.join(self.sample_path,
                                    '{}_clean.wav'.format(name[0])),
                                    clean, 16000)
                # enhanced = enhanced / enhanced.max() * 0.5


            self.validation_dataloader.desc = 'val[{}/{}]'.format(
                epoch, self.epochs + self.start_epoch-1)

            self.validation_dataloader.postfix = 'ls{:.2f},est{:.2f}'.format(
                total_loss / step, total_stoi_score / step)

        # 等待所有进程计算完毕
        if self.device != torch.device("cpu"):
            torch.cuda.synchronize(self.device)

        if self.rank == 1:
            self.writer.add_scalars(
                'val_loss', {'val_loss': total_loss / step, 
                             'estoi': total_stoi_score / step}, epoch)

        return total_loss / step, total_stoi_score / step
        
    torch.cuda.empty_cache()

    def train(self):
        if self.rank == 1:
            timestamp_txt = os.path.join(self.exp_path, 'timestamp.txt')
            mode = 'a' if os.path.exists(timestamp_txt) else 'w'
            with open(timestamp_txt, mode) as f:
                f.write('[{}] start for {} epochs\n'.format(
                    datetime.now().strftime("%Y-%m-%d-%H:%M"), self.epochs))

        if self.resume:
            self._resume_checkpoint()

        for epoch in range(self.start_epoch, self.epochs + self.start_epoch):
            self.train_sampler.set_epoch(epoch)

            self._set_train_mode()
            self._train_epoch(epoch)

            self._set_eval_mode()
            valid_loss, score = self._validation_epoch(epoch)

            #self.scheduler.step(valid_loss)

            if (self.rank == 1) and (epoch % self.save_checkpoint_interval == 0):
                self._save_checkpoint(epoch, score)

        if self.rank == 1:
            torch.save(self.state_dict_best,
                    os.path.join(self.checkpoint_path,
                    'best_model_{}.tar'.format(str(self.state_dict_best['epoch']).zfill(4))))    

            print('------------Training for {} epochs has done!------------'.format(self.epochs))

            with open(timestamp_txt, 'a') as f:
                f.write('[{}] end\n'.format(datetime.now().strftime("%Y-%m-%d-%H:%M")))

        
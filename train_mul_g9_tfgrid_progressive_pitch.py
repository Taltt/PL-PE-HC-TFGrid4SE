'''
multi gpu version
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3,4,5,6,7,2,0,1" 
import toml
import torch
import torch.nn as nn
import argparse
import torch.distributed as dist
from collections import OrderedDict
from trainer_mul_g9_tfgrid_progressive_pitch import Trainer
from SISO_TFGridNet_progressive import TFGridNet0, TFGridNetPret, TFGridNetMerge2, TFGridNetPE, TFGridNetPEPret
from Dataloader_d_g8_progressive import Dataset
from binaural_loss import loss_MTFAA_t,loss_wavmag, loss_MTFAA_fonly, loss_MTFAA_fonly_asym, loss_MTFAA_t_progressive
from pitch_estimation.pitch_estimator_model import pitch_estimator
from pitch_estimation.comb_filter import comb_filter

def run(rank,config,args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12368'
    dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
    torch.cuda.set_device(rank)
    dist.barrier()

    args.rank = rank
    args.device = torch.device(rank)

    train_dataset = Dataset(**config['train_dataset'])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, sampler=train_sampler,
                                                    **config['train_dataloader'])
    
    validation_dataset = Dataset(**config['validation_dataset'])
    validation_sampler = torch.utils.data.distributed.DistributedSampler(validation_dataset)
    validation_dataloader = torch.utils.data.DataLoader(dataset=validation_dataset, sampler=validation_sampler,
                                                        **config['validation_dataloader'])

    model_pl = TFGridNetPE()
    model_pret = TFGridNetPEPret()
    model_pitch = pitch_estimator()
    model_comb = comb_filter()
    model = TFGridNetMerge2()
    
    # Clone model parameters and buffers
    for name, param in model_comb.named_parameters():
        cloned_param = param.data.clone().detach().to(args.device)
        # exec("model_comb."+name+"=nn.Parameter(cloned_param)")
        model_comb.name = nn.Parameter(cloned_param)
        if param._grad is not None:
            #param._grad.data = param._grad.data.clone().detach().to(args.device)
            model_comb.name._grad = param._grad.clone().detach().to(args.device)

    for name, buffer in model_comb.named_buffers():
        cloned_buffer = buffer.data.clone().detach().to(args.device)
        model_comb.name = cloned_buffer

    # 模型初始化
    ckpoints = '/data/ssd1/tong.lei/exp_FTJNF/SISO_TFgrid_PE_Progressive_allnoise2_2024-07-12-23h19m/checkpoints/model_0037.tar'
    checkpoint = torch.load(ckpoints, map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in checkpoint['model'].items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v #新字典的key值对应的value为一一对应的值。
    model_pl.load_state_dict(new_state_dict)#(checkpoint['model'])# g6 dont change
    
    model_pret.conv.load_state_dict(model_pl.conv.state_dict())
    model_pret.pe.load_state_dict(model_pl.pe.state_dict())
    for x in range(model_pret.n_layers):
        model_pret.blocks[x].load_state_dict(model_pl.blocks[x].state_dict())
        model_pret.deconvs[x].load_state_dict(model_pl.deconvs[x].state_dict())
        
    checkpoint_path = '/data/ssd1/tong.lei/exp_FTJNF/Pitch_estimator/chkpt/epoch_12_trainloss_0.061130986_validloss_0.061626785.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model_pitch.load_state_dict(new_state_dict) 
    
    model.block.load_state_dict(model_pl.blocks[-1].state_dict())
    model.deconv.load_state_dict(model_pl.deconvs[-1].state_dict())
    
    model_pret.to(args.device)
    model_pitch.to(args.device)
    model_comb.to(args.device)
    model_comb.window = model_comb.window.clone().detach().to(args.device)
    # model_pret.eval()
    # model_pitch.eval()
    # model_comb.eval()   
    model.to(args.device)

    # 转为DDP模型
    model_pret = torch.nn.parallel.DistributedDataParallel(model_pret, device_ids=[rank])
    model_pitch = torch.nn.parallel.DistributedDataParallel(model_pitch, device_ids=[rank])
    model_comb = torch.nn.parallel.DistributedDataParallel(model_comb, device_ids=[rank])
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    #optimizer = torch.optim.Adam(params=model.parameters(), lr=config['optimizer']['lr'])
    optimizer = NoamOpt(model_size=config['network_config']['model_size'], factor=0.8, warmup=3000,
                    optimizer=torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))#一般warmup=step左右

    if config['loss']['loss_func'] == 'MTFAA':
        loss = loss_MTFAA_t()
    elif config['loss']['loss_func'] == 'wavmag':
        loss = loss_wavmag()
    else:
        raise NotImplementedError

    trainer = Trainer(config=config, model=[model_pret, model_pitch, model_comb, model],optimizer=optimizer, loss_func=loss,
                      train_dataloader=train_dataloader, validation_dataloader=validation_dataloader, 
                      train_sampler=train_sampler,args=args)

    trainer.train()
    dist.destroy_process_group()

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 1
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        if self._step == 2000000000:
            self._step = 1
            #self.factor = self.factor * 0.9
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))#step = warmnp时，两个相等

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', '--config', default='den_g9_tfgrid.toml')

    args = parser.parse_args()

    config = toml.load(args.config)
    args.world_size = config['DDP']['world_size']
    torch.multiprocessing.spawn(
        run, args=(config, args,), nprocs=args.world_size, join=True)


'''
multi gpu version

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1" 
import toml
import torch
import argparse
import torch.distributed as dist

from trainer_mul_g9_tfgrid_1 import Trainer
from MISO_TFGridNet import TFGridNet_nostd
from datasets_g9 import dataset_train_t,dataset_testaudio_t
from binaural_loss import loss_MTFAA_t,loss_wavmag


def run(rank,config,args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
    torch.cuda.set_device(rank)
    dist.barrier()

    args.rank = rank
    args.device = torch.device(rank)

    train_dataset = dataset_train_t(**config['train_dataset'])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, sampler=train_sampler,
                                                    **config['train_dataloader'])
    
    validation_dataset = dataset_testaudio_t(**config['validation_dataset'])
    validation_sampler = torch.utils.data.distributed.DistributedSampler(validation_dataset)
    validation_dataloader = torch.utils.data.DataLoader(dataset=validation_dataset, sampler=validation_sampler,
                                                        **config['validation_dataloader'])

    model = TFGridNet_nostd()
    model.to(args.device)

    # 转为DDP模型
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    #optimizer = torch.optim.Adam(params=model.parameters(), lr=config['optimizer']['lr'])
    optimizer = NoamOpt(model_size=config['network_config']['model_size'], factor=1.0, warmup=20000,
                    optimizer=torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))#一般warmup=step左右

    if config['loss']['loss_func'] == 'MTFAA':
        loss = loss_MTFAA_t()
    elif config['loss']['loss_func'] == 'wavmag':
        loss = loss_wavmag()
    else:
        raise NotImplementedError

    trainer = Trainer(config=config, model=model,optimizer=optimizer, loss_func=loss,
                      train_dataloader=train_dataloader, validation_dataloader=validation_dataloader, 
                      train_sampler=train_sampler,args=args)

    trainer.train()
    dist.destroy_process_group()

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 4000
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        if self._step == 120000:
            self._step = 1
            #self.factor = self.factor * 0.9
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))#step = warmnp时，两个相等

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', '--config', default='den_g9_tfgrid_1.toml')

    args = parser.parse_args()

    config = toml.load(args.config)
    args.world_size = config['DDP']['world_size']
    torch.multiprocessing.spawn(
        run, args=(config, args,), nprocs=args.world_size, join=True)
'''

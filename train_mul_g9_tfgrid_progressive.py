'''
multi gpu version
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3" 
import toml
import torch
import argparse
import torch.distributed as dist
from collections import OrderedDict
from trainer_mul_g9_tfgrid_progressive import Trainer
from SISO_TFGridNet_progressive import TFGridNet0, TFGridNetCFB, TFGridNetPE
#from SISO_SPMamba_progressive import SPMamba
# from Lite.tfgrid_intraTCN_PL import TFGridNet128 as TFGridNet
# from Dataloader_d_g8_progressive_add_babble import Dataset
from Dataloader_d_g8_progressive import Dataset
# from Dataloader_DNS3_g8_progressive import Dataset
from binaural_loss import loss_MTFAA_t,loss_wavmag, loss_MTFAA_fonly, loss_MTFAA_fonly_asym, loss_MTFAA_t_progressive


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

    model = TFGridNetPE()
    
    ### 模型初始化
    ckpoints = '/data/ssd1/tong.lei/exp_FTJNF/SISO_TFgrid_PE_Progressive_26_2023-07-04-15h06m_fulldata/checkpoints/model_0215.tar'
    checkpoint = torch.load(ckpoints, map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in checkpoint['model'].items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v #新字典的key值对应的value为一一对应的值。

    model.load_state_dict(new_state_dict)#(checkpoint['model'])# g6 dont change
    
    model.to(args.device)

    # 转为DDP模型
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    #optimizer = torch.optim.Adam(params=model.parameters(), lr=config['optimizer']['lr'])
    optimizer = NoamOpt(model_size=config['network_config']['model_size'], factor=0.1, warmup=10000,
                    optimizer=torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))#一般warmup=step左右

    if config['loss']['loss_func'] == 'MTFAA':
        loss = loss_MTFAA_t_progressive()
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
        self._step = 10
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        if self._step == 20000000000:
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

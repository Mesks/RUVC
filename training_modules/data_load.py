import os, random, sys
import cv2
import torch
import threading
import functools
import torchvision.transforms as transforms
import torch.distributed as dist
from . import configure as cfg
from auxiliary_modules import Guassian as gua
from auxiliary_modules import video_tensor_processor as vtp
from torch.utils.data import Dataset, DataLoader
        
class TrainDataset(Dataset):
    def __init__(self, training_config:cfg.Configuration, rank, world_size):
        self.ori_sample_list      = []
        self.data_index           = []
        self.sample_size          = training_config.sample_size
        self.rank                 = rank
        self.world_size           = world_size
        self.data_path            = training_config.dataset
        self.split_coefficient    = training_config.trainingset_split_coefficient
        self.GOP_size             = training_config.RUVC_GOP + 1
        self.batch_size           = training_config.batch_size
        self.random_seed          = training_config.random_seed
        self.epoch                = 0
        self.each_rank_num        = 0
        
        video_list  = [folder for folder in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, folder))]
        for video in video_list:
            GOP_list  = []
            video_dir = os.path.join(self.data_path, video)
            frames    = sorted(os.listdir(video_dir), key=lambda x: (x.isdigit(), x.lower()))
            frames    = [f for f in frames if os.path.isfile(os.path.join(video_dir, f))]
            for frame in frames:
                if len(GOP_list) < self.GOP_size:
                    GOP_list.append(os.path.join(video,frame))
                else:
                    break
            self.ori_sample_list.append(GOP_list)
                
        self.data_index = self.ori_sample_list[:int(len(self.ori_sample_list)*self.split_coefficient)]
        for i in range(3):
            if isinstance(self.random_seed, int):
                random.seed(self.random_seed)
            random.shuffle(self.data_index)
            
        self.each_rank_num = len(self.ori_sample_list)//self.world_size
        self.data_index    = self.data_index[self.each_rank_num*self.rank:self.each_rank_num*(self.rank+1)]
        discard_GPU_split  = len(self.ori_sample_list)%self.world_size
        if discard_GPU_split > 0:
            print(f"{discard_GPU_split} samples were discarded because they could not be evenly divided to the GPU.")
                        
    
    def __getitem__(self, idx) -> torch.Tensor:
        transf = transforms.ToTensor()
        sample = []
        for frame_index in self.data_index[idx]:
            frame = transf(cv2.cvtColor(cv2.imread(os.path.join(self.data_path, frame_index)), cv2.COLOR_BGR2RGB))        # (h,w,c)->(c,h,w) BGR->RGB
            sample.append(frame)
            
        sample = torch.cat(sample, dim=0).unsqueeze(0)
            
        now_seed = self.random_seed+(self.epoch+1) * idx + self.rank
        if self.sample_size>0 and self.sample_size<sample.shape[2] and self.sample_size<sample.shape[3]:
            random.seed(now_seed)
            rnd_h = random.randint(0, max(0, sample.shape[2] - self.sample_size))
            random.seed(now_seed+1)
            rnd_w = random.randint(0, max(0, sample.shape[3] - self.sample_size))
            sample = sample[:,:,rnd_h:rnd_h + self.sample_size, rnd_w:rnd_w + self.sample_size]
            
        random.seed(now_seed+2)
        sample = torch.flip(sample, dims=[3]) if random.random() < 0.5 else sample
        random.seed(now_seed+3)
        sample = torch.flip(sample, dims=[2]) if random.random() < 0.5 else sample
        random.seed(now_seed+4)
        sample = torch.rot90(sample, k=random.randint(0, 3), dims=(2, 3)) if sample.shape[2] == sample.shape[3] \
            else torch.rot90(sample, k=random.choice([0, 2]), dims=(2, 3))

        return sample.squeeze(0)
        
    def reset(self, epoch=0) -> None:
        self.epoch      = epoch
        self.data_index = self.ori_sample_list[:int(len(self.ori_sample_list)*self.split_coefficient)]
        for i in range(3):
            if isinstance(self.random_seed, int):
                random.seed(self.random_seed + epoch)
            random.shuffle(self.data_index)
                
        self.data_index = self.data_index[self.each_rank_num*self.rank:self.each_rank_num*(self.rank+1)]        
        
    def __len__(self) -> int:
        return len(self.data_index)
    
    
class EvalDataset(Dataset):
    def __init__(self, training_config:cfg.Configuration, rank, world_size):
        self.data_index           = []
        self.rank                 = rank
        self.world_size           = world_size
        self.data_path            = training_config.external_evalutionset if training_config.external_evalutionset!=None else training_config.dataset
        self.split_coefficient    = training_config.trainingset_split_coefficient
        self.GOP_size             = training_config.RUVC_GOP + 1
        self.sample_num           = 0
        
        video_list  = [folder for folder in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, folder))]
        for video in video_list:
            GOP_list  = []
            video_dir = os.path.join(self.data_path, video)
            frames    = sorted(os.listdir(video_dir), key=lambda x: (x.isdigit(), x.lower()))
            frames    = [f for f in frames if os.path.isfile(os.path.join(video_dir, f))]
            for frame in frames:
                if len(GOP_list) < self.GOP_size:
                    GOP_list.append(os.path.join(video,frame))
                else:
                    break
            self.data_index.append(GOP_list)
                                                        
        if training_config.external_evalutionset == None:
            self.data_index = self.data_index[int(len(self.data_index)*self.split_coefficient):]
            
        self.sample_num = len(self.data_index)
        
    def __getitem__(self, idx) -> torch.Tensor:
        transf = transforms.ToTensor()
        sample = []
        for frame_index in self.data_index[idx]:
            frame = transf(cv2.cvtColor(cv2.imread(os.path.join(self.data_path, frame_index)), cv2.COLOR_BGR2RGB))        # (h,w,c)->(c,h,w) BGR->RGB
            sample.append(frame)
            
        sample = torch.cat(sample, dim=0)
        
        return sample
        
    def __len__(self) -> int:
        return len(self.data_index)
    
    
class DatasetManager():
    def __init__(self, training_config:cfg.Configuration, rank:int, world_size:int):
        if rank == 0: # Multithreading happens only once
            print(f"\n====================>>>{'Data Loading'.center(25)}<<<====================")
            
        self.split_factor = training_config.trainingset_split_coefficient
            
        self.train_dataset    = TrainDataset(training_config, rank, world_size)
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=training_config.batch_size,
            num_workers=training_config.num_workers,
            shuffle=False,      # training dataset will call reset function for shuffle at the end of each epoch.
            drop_last=False
        )
        
        self.eval_dataset    = EvalDataset(training_config, rank, world_size)
        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=training_config.eval_batch_size,
            num_workers=training_config.num_workers,
            shuffle=False,
            drop_last=False
        )
        
        if rank == 0: # Multithreading happens only once
            if training_config.external_evalutionset != None:
                print(f"Adopt Training set \"{training_config.dataset}\".")
                print(f"Adopt external Evaluation set \"{training_config.external_evalutionset}\".")
            else:
                print(f"The data set \"{training_config.dataset}\" is used.")
                print(f"The ratio of the training set to the evaluation set is \"{self.split_factor}:{1-self.split_factor}\".")
                
            print(f"Trainingset contain {self.train_dataset.each_rank_num*world_size} samples.")
            print(f"Evaluationset contain {self.eval_dataset.sample_num} samples.")
            print(f"Every {training_config.RUVC_GOP+1} frames constitutes a video as a training sample.")

    def get_dataloader(self):
        return self.train_dataloader, self.eval_dataloader
    
    def new_epoch(self, epoch:int):
        self.train_dataset.reset(epoch=epoch)
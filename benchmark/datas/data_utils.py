from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
from torchnet.meter import AUCMeter
from .randaugment import RandAugmentMC
import copy
import sys


def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

def get_imbalance_ratios_1(imb_factor, cls_num):
    imbalance_ratios = []
    for cls_idx in range(cls_num):
        ratio = imb_factor ** (cls_idx / (cls_num - 1.0))
        imbalance_ratios.append(ratio)
    return imbalance_ratios

def get_img_num_per_cls_1(dataset,cls_num,imb_factor=None,num_meta=None):
    """
    Get a list of image numbers for each class, given cifar version
    Num of imgs follows emponential distribution
    img max: 5000 / 500 * e^(-lambda * 0);
    img min: 5000 / 500 * e^(-lambda * int(cifar_version - 1))
    exp(-lambda * (int(cifar_version) - 1)) = img_max / img_min
    args:
      cifar_version: str, '10', '100', '20'
      imb_factor: float, imbalance factor: img_min/img_max,
        None if geting default cifar data number
    output:
      img_num_per_cls: a list of number of images per class
    """
    if dataset == 'cifar10':
        img_max = 5000-num_meta

    if dataset == 'cifar100':
        img_max = 500-num_meta

    if imb_factor is None:
        return [img_max] * cls_num
    img_num_per_cls = []
    imbalance_ratios = get_imbalance_ratios_1(imb_factor, cls_num)
    for cls_idx in range(cls_num):
        ratio = imbalance_ratios[cls_idx]
        num = img_max * ratio
        img_num_per_cls.append(int(num))
    return img_num_per_cls

def sample_dataset_1(train_data,train_label, img_num_list, num_classes, kind):
    """
    Args:
        dataset
        img_num_list
        num_classes
        kind
    Returns:

    
    """
    data_list = {}
    for j in range(num_classes):
        data_list[j] = [i for i, label in enumerate(train_label) if label == j]

    idx_to_del = []
    for cls_idx, img_id_list in data_list.items():
        '''
        cls_idx : class index
        img_id_list:sample global index list
        data_list:{'cls_idx':[img_id_list],}
        '''
        np.random.shuffle(img_id_list)
        img_num = img_num_list[int(cls_idx)]
        if kind=='delete':
            idx_to_del.extend(img_id_list[:img_num])
        else:
            idx_to_del.extend(img_id_list[img_num:])

    # new_dataset = copy.deepcopy(dataset)
    train_label = np.delete(train_label, idx_to_del, axis=0)
    train_data = np.delete(train_data, idx_to_del, axis=0)
    # data_index = np.delete(data_index, idx_to_del, axis=0)
    return train_data,train_label

def uniform_mix_c_1(mixing_ratio, num_classes):
    """
    returns a linear interpolation of a uniform matrix and an identity matrix
    """
    return mixing_ratio * np.full((num_classes, num_classes), 1 / num_classes) + \
        (1 - mixing_ratio) * np.eye(num_classes)

def flip_labels_c_1(corruption_prob, num_classes, seed=1):
    """
    returns a matrix with (1 - corruption_prob) on the diagonals, and corruption_prob
    concentrated in only one other entry for each row
    """
    np.random.seed(seed)
    C = np.eye(num_classes) * (1 - corruption_prob)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        C[i][np.random.choice(row_indices[row_indices != i])] = corruption_prob

    return C

class cifar_dataset(Dataset): 
    def __init__(self, dataset, r, noise_mode, root_dir, transform, mode, noise_file='', select_index=[], probability=[], log='', imb_factor=None, seed=0, transform_strong=None): 
        np.random.seed(seed)
        random.seed(seed)
        self.r = r # noise ratio
        self.transform = transform
        self.transform_strong = transform_strong
        self.mode = mode  
        self.probability = np.array(probability)

        if self.mode=='test':
            if dataset=='cifar10':
                test_dic = unpickle('%s/test_batch'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['labels']
            elif dataset=='cifar100':
                test_dic = unpickle('%s/test'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['fine_labels']                            
        else:    
            train_data=[]
            train_label=[]
            if dataset=='cifar10': 
                for n in range(1,6):
                    num_class =10
                    dpath = '%s/data_batch_%d'%(root_dir,n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label+data_dic['labels']
                train_data = np.concatenate(train_data)
            elif dataset=='cifar100':    
                num_class =100
                train_dic = unpickle('%s/train'%root_dir)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))
            
            # imbalance
            img_num_list = get_img_num_per_cls_1(dataset, num_class, imb_factor, 0)
            self.img_num_list = img_num_list
            train_data ,train_label=sample_dataset_1(train_data ,train_label,img_num_list,num_class,'select')
            num_sample = sum(img_num_list)
            
            # noisy
            if noise_mode=='unif':
                noisy_transaction_matrix_real = uniform_mix_c_1(self.r, num_class)
            elif noise_mode=='flip':
                noisy_transaction_matrix_real = flip_labels_c_1(self.r, num_class)
            noise_label = copy.deepcopy(train_label)
            self.train_label = train_label
            for i in range(sum(img_num_list) ):
                noise_label[i] = np.random.choice(num_class, p=noisy_transaction_matrix_real[train_label[i]])     
            
            if self.mode == 'all' or self.mode == 'coteaching_semi':
                self.train_data = train_data
                self.noise_label = noise_label
            else:
                if self.mode == "labeled" or self.mode == "supervised" or self.mode=="labeled_fixmatch":
                    self.train_data = train_data[select_index]
                    self.noise_label = noise_label[select_index]
                    
                elif self.mode == "unlabeled" or self.mode=="supervised_unlabeled" or self.mode=="unlabeled_fixmatch":
                    self.train_data = np.delete(train_data,select_index ,axis=0)
                    self.noise_label = np.delete(noise_label,select_index ,axis=0) 
                
    def __getitem__(self, index):
        if self.mode=='labeled':
            img, target, prob = self.train_data[index], self.noise_label[index], self.probability[index]
            img = Image.fromarray(img)
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            return img1, img2, target, prob
        elif self.mode=='supervised':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, img2, target
        if self.mode=='labeled_fixmatch':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img1 = self.transform(img) 
            return img1, target
        elif self.mode=='supervised_unlabeled':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img1 = self.transform(img) 
            return img1, target
        elif self.mode=='unlabeled':
            img = self.train_data[index]
            img = Image.fromarray(img)
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            return img1, img2
        elif self.mode=='unlabeled_fixmatch':
            img = self.train_data[index]
            img = Image.fromarray(img)
            img1 = self.transform(img) 
            img2 = self.transform_strong(img)
            return img1, img2        
        elif self.mode=='all':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target, index
        elif self.mode=='coteaching_semi':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, img2, target, index
        elif self.mode=='test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_data)
        else:
            return len(self.test_data)         
        
        
class cifar_dataloader():  
    def __init__(self, dataset, r, imb_factor, noise_mode, batch_size, num_workers, root_dir, log, noise_file='', seed=0):
        self.dataset = dataset
        self.r = r
        self.seed = seed
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.noise_file = noise_file
        self.imb_factor = imb_factor
        if self.dataset=='cifar10':
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ])
            
            self.transform_train_strong = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    RandAugmentMC(n=2, m=10),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ])
            
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ])    
        elif self.dataset=='cifar100':    
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]) 
            
            self.transform_train_strong = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    RandAugmentMC(n=2, m=10),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])             
            
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])   
    def run(self,mode,select_index=[],prob=[]):
        if mode=='warmup':
            all_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, imb_factor=self.imb_factor, root_dir=self.root_dir, transform=self.transform_train, mode="all",noise_file=self.noise_file, seed=self.seed)
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)
            return trainloader
        
        if mode=='coteaching_semi':
            all_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, imb_factor=self.imb_factor, root_dir=self.root_dir, transform=self.transform_train, mode="coteaching_semi",noise_file=self.noise_file, seed=self.seed)
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)
            return trainloader        
        
        elif mode=='warmup_dataset':
            all_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, imb_factor=self.imb_factor, root_dir=self.root_dir, transform=self.transform_train, mode="all",noise_file=self.noise_file, seed=self.seed)
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)             
            return trainloader, all_dataset    
        
        elif mode=='selection_label':
            all_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, imb_factor=self.imb_factor, root_dir=self.root_dir, transform=self.transform_train, mode="all",noise_file=self.noise_file, seed=self.seed)
            
            return all_dataset.train_label, all_dataset.noise_label
                                     
        elif mode=='train':
            labeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, imb_factor=self.imb_factor, root_dir=self.root_dir, transform=self.transform_train, mode="labeled", noise_file=self.noise_file, select_index=select_index, probability=prob,log=self.log, seed=self.seed)
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)   
            
            unlabeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, imb_factor=self.imb_factor, root_dir=self.root_dir, transform=self.transform_train, mode="unlabeled", noise_file=self.noise_file, select_index=select_index, seed=self.seed)
            if len(unlabeled_dataset.noise_label)==0:
                return labeled_trainloader, None
            
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)     
            return labeled_trainloader, unlabeled_trainloader
        
        elif mode=='train_fixmatch':
            labeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, imb_factor=self.imb_factor, root_dir=self.root_dir, transform=self.transform_train, mode="labeled_fixmatch", noise_file=self.noise_file, select_index=select_index, probability=prob,log=self.log, seed=self.seed)
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)   
            
            unlabeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, imb_factor=self.imb_factor, root_dir=self.root_dir, transform=self.transform_train, mode="unlabeled_fixmatch", noise_file=self.noise_file, select_index=select_index, seed=self.seed, transform_strong=self.transform_train_strong)
            if len(unlabeled_dataset.noise_label)==0:
                return labeled_trainloader, None
            
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)     
            return labeled_trainloader, unlabeled_trainloader        
        
        elif mode=='supervised':
            labeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, imb_factor=self.imb_factor, root_dir=self.root_dir, transform=self.transform_train, mode="supervised", noise_file=self.noise_file, select_index=select_index, probability=prob,log=self.log, seed=self.seed)
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)   
            return labeled_trainloader
        
        elif mode=='supervised_with_unlabeled':
            labeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, imb_factor=self.imb_factor, root_dir=self.root_dir, transform=self.transform_train, mode="supervised", noise_file=self.noise_file, select_index=select_index, probability=prob,log=self.log, seed=self.seed)
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
            
            unlabeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, imb_factor=self.imb_factor, root_dir=self.root_dir, transform=self.transform_train, mode="supervised_unlabeled", noise_file=self.noise_file, select_index=select_index, seed=self.seed)
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
            
            return labeled_trainloader, unlabeled_trainloader
        
        elif mode=='train_dataset':
            labeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, imb_factor=self.imb_factor, root_dir=self.root_dir, transform=self.transform_train, mode="labeled", noise_file=self.noise_file, select_index=select_index, probability=prob,log=self.log, seed=self.seed)
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)   
            
            unlabeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, imb_factor=self.imb_factor, root_dir=self.root_dir, transform=self.transform_train, mode="unlabeled", noise_file=self.noise_file, select_index=select_index, seed=self.seed)
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)     
            return labeled_trainloader, unlabeled_trainloader, labeled_dataset
        
        elif mode=='test':
            test_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='test', seed=self.seed)
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return test_loader
        
        elif mode=='eval_train' or mode=='all':
            eval_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, imb_factor=self.imb_factor, root_dir=self.root_dir, transform=self.transform_test, mode='all', noise_file=self.noise_file, seed=self.seed)
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return eval_loader        
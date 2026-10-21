from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import torch
import os, sys
   

def get_imbalance_ratios(imb_factor, cls_num=10):
    imbalance_ratios = []
    for cls_idx in range(cls_num):
        ratio = imb_factor ** (cls_idx / (cls_num - 1.0))
        imbalance_ratios.append(ratio)
    return imbalance_ratios

class animal_dataset(Dataset): 
    def __init__(self, root, transform, mode, pred=[], probability=[], paths=[], imb_factor=None):
        
        self.root = root
        self.mode = mode
        self.train_dir = os.path.join(self.root, 'training')
        self.test_dir = os.path.join(self.root, 'testing')
        self.clsList = sorted(os.listdir(self.train_dir))
        self.nbCls = len(self.clsList)
        self.cls2Idx = dict(zip(self.clsList, range(self.nbCls)))
        self.imb_factor = imb_factor
        self.transform = transform
        
        if mode=='test':
            self.test_imgs = []
            self.test_labels = []
            for cls in self.clsList:
                test_img_list = sorted(os.listdir(os.path.join(self.test_dir, str(cls))))
                self.test_imgs = self.test_imgs + [os.path.join(self.test_dir, str(cls), img) for img in test_img_list]
                self.test_labels = self.test_labels + [self.cls2Idx[cls] for i in range(len(test_img_list))]            
        else:
            img_max = 5466 # number of class 0 samples
            img_num_per_cls = []
            imbalance_ratios = get_imbalance_ratios(self.imb_factor)
            for cls in range(10):
                ratio = imbalance_ratios[cls]
                num = int(img_max * ratio)
                img_num_per_cls.append(int(num))
            self.total_samples = sum(img_num_per_cls)
            
            train_imgs = []
            train_labels = []
            
            # train original: {0:5466, 1:4608, 2:5091, 3:4841, 4:4981, 5:4913, 6:5322, 7:4999, 8:4970, 9:4809}
            # test original: {0:557, 1:485, 2:423, 3:410, 4:509, 5:524, 6:620, 7:557, 8:440, 9:475}
            for cls in self.clsList:
                train_img_list = sorted(os.listdir(os.path.join(self.train_dir, str(cls))))
                train_imgs = train_imgs + [os.path.join(self.train_dir, str(cls), img) for img in train_img_list][:img_num_per_cls[int(cls)]]
                train_labels = train_labels + [self.cls2Idx[cls] for i in range(len(train_img_list))][:img_num_per_cls[int(cls)]]            
            
            if self.mode == 'all' or self.mode=='coteaching_semi':
                self.train_imgs = train_imgs
                self.train_labels = train_labels
            elif self.mode == 'labeled':
                pred_idx = pred.nonzero()[0]
                # self.train_imgs = train_imgs[pred_idx]
                # self.probability = probability[pred_idx]
                self.train_imgs = [train_imgs[i] for i in pred_idx]
                self.train_labels = [train_labels[i] for i in pred_idx]
                self.probability = [probability[i] for i in pred_idx]     
                print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))
                
            elif self.mode == "unlabeled":  
                pred_idx = (1-pred).nonzero()[0]  
                self.train_imgs = [train_imgs[i] for i in pred_idx]
                self.train_labels = [train_labels[i] for i in pred_idx]
                # self.probability = [probability[i] for i in pred_idx]     
                print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))
                    
    def __getitem__(self, index):  
        if self.mode=='labeled':
            img_path = self.train_imgs[index]
            target = self.train_labels[index]
            prob = self.probability[index]
            image = Image.open(img_path).convert('RGB')
            img1 = self.transform(image) 
            img2 = self.transform(image) 
            return img1, img2, target, prob              
        elif self.mode=='unlabeled':
            img_path = self.train_imgs[index]
            image = Image.open(img_path).convert('RGB')
            img1 = self.transform(image) 
            img2 = self.transform(image) 
            return img1, img2  
        elif self.mode=='all':
            img_path = self.train_imgs[index]
            target = self.train_labels[index]     
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target, img_path
        elif self.mode=='coteaching_semi':
            img_path = self.train_imgs[index]
            target = self.train_labels[index]
            image = Image.open(img_path).convert('RGB')
            img1 = self.transform(image) 
            img2 = self.transform(image) 
            return img1, img2, target, index       
        elif self.mode=='test':
            img_path = self.test_imgs[index]
            target = self.test_labels[index]     
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image) 
            return img, target   
        
    def __len__(self):
        if self.mode=='test':
            return len(self.test_imgs)
        else:
            return len(self.train_imgs)       
        
class animal_dataloader():
    def __init__(self, root, batch_size, num_workers, imb_factor):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root = root
        self.imb_factor = imb_factor

        self.transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
         ])

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])    
    def run(self,mode,pred=[],prob=[],paths=[]):        
        if mode=='warmup':
            warmup_dataset = animal_dataset(self.root,transform=self.transform_train, mode='all', imb_factor=self.imb_factor)
            warmup_loader = DataLoader(
                dataset=warmup_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)  
            return warmup_loader
        
        elif mode=='coteaching_semi':
            warmup_dataset = animal_dataset(self.root,transform=self.transform_train, mode='coteaching_semi', imb_factor=self.imb_factor)
            warmup_loader = DataLoader(
                dataset=warmup_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)  
            return warmup_loader
        
        elif mode=='train':
            labeled_dataset = animal_dataset(self.root,transform=self.transform_train, mode='labeled',pred=pred, probability=prob,paths=paths, imb_factor=self.imb_factor)
            labeled_loader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)           
            unlabeled_dataset = animal_dataset(self.root,transform=self.transform_train, mode='unlabeled',pred=pred, probability=prob,paths=paths, imb_factor=self.imb_factor)
            unlabeled_loader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=int(self.batch_size),
                shuffle=True,
                num_workers=self.num_workers)   
            return labeled_loader,unlabeled_loader
        elif mode=='eval_train':
            eval_dataset = animal_dataset(self.root,transform=self.transform_test, mode='all', imb_factor=self.imb_factor)
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return eval_loader, eval_dataset.total_samples
            
        elif mode=='test':
            test_dataset = animal_dataset(self.root,transform=self.transform_test, mode='test')
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size*2,
                shuffle=False,
                num_workers=self.num_workers)             
            return test_loader
        
        elif mode=='selection_label':
            eval_dataset = animal_dataset(self.root,transform=self.transform_test, mode='all', imb_factor=self.imb_factor)
            return np.array(eval_dataset.train_labels)
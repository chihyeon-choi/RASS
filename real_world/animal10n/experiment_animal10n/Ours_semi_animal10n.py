from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.models as models
import random
import os
import copy
import argparse
import numpy as np
import dataloader_animal10n as dataloader
from model_vgg_feature import vgg19_bn
from sklearn.mixture import GaussianMixture
import pandas as pd

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

parser = argparse.ArgumentParser(description='PyTorch Animal10N Training')
parser.add_argument('--batch_size', default=32, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.002, type=float, help='initial learning rate')
parser.add_argument('--lambda_u', default=0, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--id', default='Animal10N')
parser.add_argument('--data_path', default='../datasets/Animal10N/raw_image', type=str, help='path to dataset')
parser.add_argument('--seed', default=123, type=int)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--num_batches', default=1000, type=int)
parser.add_argument('--imb_factor', default=0.1, type=float)
parser.add_argument('--alpha', default=3.0, type=float)
parser.add_argument('--tau', default=0.8, type=float)
parser.add_argument('--constant_bce', default=0.1, type=float)
parser.add_argument('--warm_up', default=10, type=int)
args = parser.parse_args()

# fixed seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.cuda.set_device(args.gpuid)

save_path = os.path.join(f'./RASS_semi_Animal10N_imb{args.imb_factor}_warmup{args.warm_up}')
createDirectory(save_path)

# Training
def train(epoch,net,net2,optimizer,labeled_trainloader,unlabeled_trainloader, confident_class):
    net.train()
    net2.eval() #fix one network and train the other
    
    confident_class_tensor = torch.tensor(confident_class, dtype=torch.bool).cuda()
    unlabeled_train_iter = iter(unlabeled_trainloader) # regarded as noisy labeled data
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):      
        try:
            inputs_u, inputs_u2 = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = next(unlabeled_train_iter)
        batch_size = inputs_x.size(0)
        
        original_labels_x = copy.deepcopy(labels_x)
        original_labels_x = original_labels_x.cuda()        
        
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        # prepare for mixmatch loss
        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11, _ = net(inputs_u)
            outputs_u12, _ = net(inputs_u2)
            outputs_u21, _ = net2(inputs_u)
            outputs_u22, _ = net2(inputs_u2)            
            
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
            ptu = pu**(1/args.T) # temparature sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            targets_u = targets_u.detach()       
            
            # label refinement of labeled samples
            outputs_x, _ = net(inputs_x)
            outputs_x2, _ = net(inputs_x2)
            
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x*labels_x + (1-w_x)*px              
            ptx = px**(1/args.T) # temparature sharpening 
                       
            targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
            targets_x = targets_x.detach()       
        
        # mixmatch
        l = np.random.beta(4, 4)        
        l = max(l, 1-l)        
        
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        mixed_input = l * input_a[:batch_size*2] + (1 - l) * input_b[:batch_size*2]        
        mixed_target = l * target_a[:batch_size*2] + (1 - l) * target_b[:batch_size*2]
                
        logits, _ = net(mixed_input)
        logits_x = logits[:batch_size*2]
        
        # mixmatch loss
        Lx = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1))

        probs = torch.softmax(logits_x, dim=1)  # (batch_size, num_classes)
        
        all_targets_labeled = torch.cat([original_labels_x, original_labels_x], dim=0)
        mask = torch.ones_like(probs, dtype=torch.bool)
        mask.scatter_(1, all_targets_labeled.unsqueeze(1), False)
        probs_other = probs[mask].view(batch_size*2, -1)
        
        top_probs, _ = probs_other.max(dim=1)
        
        # negative learning is applied only to sapmles with contident_class equal to 1
        confident_class_mask = confident_class_tensor[all_targets_labeled.long()]
        if confident_class_mask.any():
            # Negative loss
            target = torch.zeros_like(top_probs) # Eq.(7) negative pseudo-label
            Lx_bce = (F.binary_cross_entropy(top_probs[confident_class_mask], target[confident_class_mask], reduction='none')).sum()/confident_class_mask.sum() # Eq.(8) negative loss
        else:
            # When there is no samples to apply negative learning, we set Lx_bce to 0 (it is no impact to training model)
            Lx_bce = torch.tensor(0.0, device=logits.device)
        
        # penalty loss
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        # Eq.(10) semi-supervised objective
        loss = Lx + penalty + Lx_bce*args.constant_bce
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write('Animal10N | Epoch [%3d/%3d] Iter[%3d/%3d]\t  Labeled loss: %.4f '
                %(epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item()))
        sys.stdout.flush()
    
def warmup(net,optimizer,dataloader):
    net.train()
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        outputs, _ = net(inputs)              
        loss = CEloss(outputs, labels)  
        
        penalty = conf_penalty(outputs)
        L = loss + penalty       
        L.backward()  
        optimizer.step() 

        sys.stdout.write('\r')
        sys.stdout.write('|Warm-up: Iter[%3d/%3d]\t CE-loss: %.4f  Conf-Penalty: %.4f'
                %(batch_idx+1, len(dataloader), loss.item(), penalty.item()))        
        sys.stdout.flush()

def test(net1,net2,test_loader):
    net1.eval()
    net2.eval()
    class_correct = {i: 0 for i in range(args.num_class)}
    class_total = {i: 0 for i in range(args.num_class)}
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1, _ = net1(inputs)       
            outputs2, _ = net2(inputs)           
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)            
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()      
            
            predicted = predicted.cpu()
            for label, pred in zip(targets, predicted):
                if label == pred:
                    class_correct[label.item()] += 1
                class_total[label.item()] += 1               
    class_accuracy = {class_idx: [class_correct[class_idx] / class_total[class_idx]] 
                       for class_idx in range(args.num_class)}  
                          
    class_accuracy_df = pd.DataFrame(class_accuracy)
    per_class_acc_dir = os.path.join(save_path, f'per_class_acc_alpha{args.alpha}_tau{args.tau}_constant{args.constant_bce}_seed{args.seed}.csv')
    if not os.path.exists(per_class_acc_dir):
        class_accuracy_df.to_csv(per_class_acc_dir, index=False, mode='w')
    else:
        class_accuracy_df.to_csv(per_class_acc_dir, index=False, mode='a', header=False)
                          
    acc = 100.*correct/total
    print("\n| Test Acc: %.2f%%\n" %(acc))  
    return acc    

def selection_or_not(cls, confidence):
    max_value, max_ind = torch.max(confidence, dim=1)
    unique_values, counts = torch.unique(max_ind, return_counts=True)
    _, max_counts_ind = torch.max(counts, dim=0)
    max_class = unique_values[max_counts_ind].item()
    
    max_value, max_ind = torch.max(confidence, dim=1)
    
    cls_max_ind = torch.where(max_ind==cls)[0]
    if len(cls_max_ind)==0: # When M_k contains no samples, we regard class k as unready class
        return False, 1000, False
    else:    
        cls_max_confidence = torch.mean(confidence[cls_max_ind], dim=0)
        sorted_confidence = cls_max_confidence[torch.argsort(cls_max_confidence)]
        highest, second_highest = sorted_confidence[-1], sorted_confidence[-2]
        if highest > second_highest * args.alpha: # Eq.(1) recognition of class readiness
            confidence_cls = confidence[:, cls]
            if max_class == cls: # if class k is ready to select & max_class (y_mode^k==k) is k, we regard such class as clean dominant class
                return True, torch.max(confidence_cls[cls_max_ind]).item()*args.tau, True 
            else:
                return True, torch.max(confidence_cls[cls_max_ind]).item()*args.tau, False
        else:
            return False, 1000, False

def eval_train(epoch,model,num_samples):
    model.eval()
    
    confidences = torch.zeros(num_samples, args.num_class)
    total_features = torch.zeros(num_samples, 128)
    cls_confidences = torch.zeros(num_samples)    
    total_labels = torch.zeros(num_samples).long()
    euc_dists = torch.zeros(num_samples)
    
    paths = []
    n=0
    with torch.no_grad():
        for batch_idx, (inputs, targets, path) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            outputs, features = model(inputs) 
            loss = CE(outputs, targets)
            confidence = torch.softmax(outputs, dim=1)
            for b in range(inputs.size(0)):
                # losses[n]=loss[b]
                confidences[n] = confidence[b]
                cls_confidences[n] = confidence[b][targets[b]]
                total_features[n] = features[b]
                total_labels[n] = targets[b]
                paths.append(path[b])
                n+=1
            sys.stdout.write('\r')
            sys.stdout.write('| Evaluating loss Iter %3d\t' %(batch_idx)) 
            sys.stdout.flush()
    
    prob = np.zeros(num_samples)
    confident_class = [0 for cls in range(args.num_class)]
    for cls in range(args.num_class):
        cls_ind = np.where(total_labels==cls)[0]
        # Below process are conducted in selection_or_not function
        # 1) Recognition of Class Readiness
        # 2) Clean-Dominant Class Identification         
        perform_selection, confidence_threshold, confident = selection_or_not(cls, confidences[cls_ind])
        if confident:
            confident_class[cls]=1
        if perform_selection:
            temp_confidences = cls_confidences[cls_ind]
            temp_criteria = torch.where(temp_confidences>=confidence_threshold)[0]  # representative sample set index
            
            high_index = cls_ind[temp_criteria]
            high_confidence_features = total_features[high_index]
            if len(temp_criteria)==1:
                high_confidence_features = high_confidence_features.unsqueeze(0)
            centroid = torch.mean(high_confidence_features, axis=0) # Eq.(3) calculation of prototype
            cls_features = total_features[cls_ind] # feature vector of individual samples in each class
            
            euclidean_dist = torch.norm(cls_features - centroid.unsqueeze(0), p=2, dim=1)  # Euclidean distance from prototype and each sample
            euc_dists[cls_ind] = euclidean_dist
            cls_euc = euc_dists[cls_ind]
            cls_euc = (cls_euc - cls_euc.min()) / (cls_euc.max()-cls_euc.min())
            cls_euc = cls_euc.reshape(-1, 1)
            
            # fit a two-component GMM to the loss
            gmm = GaussianMixture(n_components=2,max_iter=30,tol=1e-2,reg_covar=5e-4) 
            gmm.fit(cls_euc)
            cls_prob = gmm.predict_proba(cls_euc) 
            cls_prob = cls_prob[:,gmm.means_.argmin()]
            prob[cls_ind] = cls_prob
        else:
            prob[cls_ind] = np.array([1.0 for k in range(len(cls_ind))])            
         
    return prob,paths,confident_class
    
class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))
               
def create_model():
    model = vgg19_bn(num_classes=10, pretrained=False)
    model = model.cuda()
    return model      

log=open(os.path.join(save_path, f'{args.id}_alpha{args.alpha}_tau{args.tau}_constant{args.constant_bce}_{args.seed}.txt'), 'w')
log.flush()

loader = dataloader.animal_dataloader(root=args.data_path,batch_size=args.batch_size,num_workers=5,imb_factor=args.imb_factor)

print('| Building net')
net1 = create_model()
net2 = create_model()

optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
conf_penalty = NegEntropy()

best_acc = [0,0]
for epoch in range(args.num_epochs+1):   
    lr=args.lr
    if epoch >= 40:
        lr /= 10       
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr     
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr    
        
    if epoch<args.warm_up:     # warm up model
        train_loader = loader.run('warmup')
        print('Warmup Net1')
        warmup(net1,optimizer1,train_loader)     
        train_loader = loader.run('warmup')
        print('\nWarmup Net2')
        warmup(net2,optimizer2,train_loader)
    else:            
        pred1 = (prob1 > 0.5)  # returns True for 'selected as clean samples' and 'samples in each unready class'
        pred2 = (prob2 > 0.5)  # returns True for 'selected as clean samples' and 'samples in each unready class'
        
        print('\n\nTrain Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred2,prob2,paths=paths2) # co-divide
        # train net1 based on the selection results from net2
        train(epoch,net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader, confident_class2)              # train net1
        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred1,prob1,paths=paths1) # co-divide
        # train net2 based on the selection results from net1
        train(epoch,net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader, confident_class1)              # train net2
    
    test_loader = loader.run('test')    
    acc = test(net1,net2,test_loader)
    log.write('Test Accuracy:%.2f\n'%(acc))
    log.flush()    
    
    # construct sample set using each model
    print('\n==== net 1 evaluate next epoch training data loss ====') 
    eval_loader, num_samples = loader.run('eval_train')  # evaluate training data loss for next epoch  
    prob1,paths1, confident_class1 = eval_train(epoch,net1,num_samples) 
    print('\n==== net 2 evaluate next epoch training data loss ====') 
    eval_loader, num_samples = loader.run('eval_train')  
    prob2,paths2, confident_class2 = eval_train(epoch,net2,num_samples) 

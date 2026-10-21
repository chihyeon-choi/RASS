import os
import sys
from matplotlib.pyplot import get
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch
import torch.optim as optim
from sklearn.mixture import GaussianMixture
from settings.configs import *
from datas.data_utils import *
from models.PreResNet import *
import pickle
from math import inf
import string
import random
from sklearn import metrics

# get shell arg
args = parse_arguments()

# fixed seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# device detect
device = set_device(args)

# Dataloader set
kwargs = {'num_workers': 1,
          'pin_memory': True}

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

# If all classes are considered to be unready classes, we train model using supervised objective
def train_supervised(epoch,net,net2,optimizer,labeled_trainloader,unlabeled_trainloader, confident_class, bce_weight):
    net.train()
    net2.eval() #fix one network and train the other
    
    confident_class_tensor = torch.tensor(confident_class, dtype=torch.bool).cuda()
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):
        batch_size = inputs_x.size(0)

        inputs_x, labels_x = inputs_x.cuda(), labels_x.cuda()        
        logits, _ = net(inputs_x)
        
        probs = torch.softmax(logits, dim=1)  # (batch_size, num_classes)
        
        mask = torch.ones_like(probs, dtype=torch.bool)
        mask.scatter_(1, labels_x.unsqueeze(1), False)
        probs_other = probs[mask].view(batch_size, -1)
        
        top_probs, _ = probs_other.max(dim=1)

        # negative learning is applied only to sapmles with contident_class equal to 1
        confident_class_mask = confident_class_tensor[labels_x.long()]
        if confident_class_mask.any():
            # Negative loss
            target = torch.zeros_like(top_probs) # Eq.(7) negative pseudo-label
            Lx_bce = (F.binary_cross_entropy(top_probs[confident_class_mask], target[confident_class_mask], reduction='none')).sum()/confident_class_mask.sum() # Eq.(8) negative loss
        else:
            # When there is no samples to apply negative learning, we set Lx_bce to 0 (it is no impact to training model)
            Lx_bce = torch.tensor(0.0, device=logits.device)
        
        # cross-entropy loss
        Lx = F.cross_entropy(logits, labels_x)
        
        # penalty loss
        prior = torch.ones(args.num_classes)/args.num_classes
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        # Eq.(9) supervised objective
        loss = Lx + penalty + Lx_bce*args.constant_bce
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % args.print_freq == 0:
            sys.stdout.write('\r')
            sys.stdout.write('%s:%.1f-%s-%.1f | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f'
                    %(args.dataset, args.imb_factor, args.corruption_type,args.corruption_prob, epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item()))
            sys.stdout.flush()

def train(epoch,net,net2,optimizer,labeled_trainloader,unlabeled_trainloader, confident_class, bce_weight):
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
        labels_x = torch.zeros(batch_size, args.num_classes).scatter_(1, labels_x.view(-1,1), 1)
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
        
        mixed_input = l * input_a + (1 - l) * input_b        
        mixed_target = l * target_a + (1 - l) * target_b
                
        logits, _ = net(mixed_input) # [256, 10]
        logits_x = logits[:batch_size*2] # [128, 10]
        logits_u = logits[batch_size*2:] # [128, 10]
        
        # mixmatch loss
        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, warm_up)
        
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
            Lx_bce = (F.binary_cross_entropy(top_probs[confident_class_mask], target[confident_class_mask], reduction='none')).sum()/confident_class_mask.sum()  # Eq.(8) negative loss
        else:
            # When there is no samples to apply negative learning, we set Lx_bce to 0 (it is no impact to training model)
            Lx_bce = torch.tensor(0.0, device=logits.device)
        
        
        # penalty loss
        prior = torch.ones(args.num_classes)/args.num_classes
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        # Eq.(10) semi-supervised objective
        loss = Lx + lamb * Lu  + penalty*args.penalty + Lx_bce*args.constant_bce
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % args.print_freq == 0:
            sys.stdout.write('\r')
            sys.stdout.write('%s:%.1f-%s-%.1f | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'
                    %(args.dataset, args.imb_factor, args.corruption_type,args.corruption_prob, epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item(), Lu.item()))
            sys.stdout.flush()                    


def warmup_net(epoch,net,optimizer,dataloader):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1

    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        outputs, _ = net(inputs)
        
        loss = CEloss(outputs, labels)
        
        loss.backward()  
        optimizer.step() 
        if batch_idx % args.print_freq == 0:
            sys.stdout.write('\r')
            sys.stdout.write('%s:%.1f-%s-%.1f | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                    %(args.dataset, args.imb_factor, args.corruption_type,args.corruption_prob, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
            sys.stdout.flush()  

def test(epoch,net1,net2):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    class_correct = {i: 0 for i in range(args.num_classes)}
    class_total = {i: 0 for i in range(args.num_classes)}        
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
                       for class_idx in range(args.num_classes)}  
                          
    acc = 100.*correct/total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  
    test_log.write('Epoch:%d   Accuracy:%.2f\n'%(epoch,acc))
    test_log.flush()  
    
    class_accuracy_df = pd.DataFrame(class_accuracy)
    per_class_acc_dir = os.path.join(save_warmup_path, f'per_class_acc_seed{args.seed}.csv')
    if not os.path.exists(per_class_acc_dir):
        class_accuracy_df.to_csv(per_class_acc_dir, index=False, mode='w')
    else:
        class_accuracy_df.to_csv(per_class_acc_dir, index=False, mode='a', header=False)    

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        probs = torch.clamp(probs, min=1e-5)
        return torch.sum(probs.log()*probs, dim=1)
    
    
def create_model():
    model = ResNet18_feature(num_classes=args.num_classes)
    model = model.cuda()
    return model


# KL divergence
def kl_divergence(p, q):
    return (p * ((p+1e-10) / (q+1e-10)).log()).sum(dim=1)

CE = nn.CrossEntropyLoss(reduction='none')

save_warmup_path =   f'./RASS_benchmark_semi_supervised/constant_{args.constant_bce}_alpha_{args.alpha}_tau{args.tau}_{args.dataset}_imb{args.imb_factor}_{args.corruption_type}{args.corruption_prob}_penalty{args.penalty}_seed{args.seed}'
createDirectory(save_warmup_path)

stats_log=open(os.path.join(save_warmup_path, 'stats.txt'), 'w') 
test_log = open(os.path.join(save_warmup_path, 'acc.txt'), 'w')
loader = cifar_dataloader(args.dataset,r=args.corruption_prob,imb_factor=args.imb_factor,noise_mode=args.corruption_type,batch_size=args.batch_size,num_workers=5,\
    root_dir=args.data_path,log=stats_log)

clean_targets, noisy_targets = loader.run('selection_label')
num_samples = len(clean_targets)

if args.dataset=='cifar10':
    warm_up = 10
    # warm_up = 1
elif args.dataset=='cifar100':
    warm_up = 30
    # warm_up = 1

def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch,warm_up)

weight_list = np.linspace(0, 1, (args.num_epochs-warm_up+1))
weight_list = 1/np.exp(weight_list*5)

print('| Building net')
net1 = create_model()
net2 = create_model()
criterion = SemiLoss()
optimizer1 = torch.optim.SGD(net1.parameters(), lr=0.02, weight_decay=5e-4, momentum=0.9)
optimizer2 = torch.optim.SGD(net2.parameters(), lr=0.02, weight_decay=5e-4, momentum=0.9)

CEloss = nn.CrossEntropyLoss()
conf_penalty = NegEntropy()
test_loader = loader.run('test')

warm_up_flag = True
eval_flag = False

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


def eval_train(model, all_loss, all_jsd, all_entropy, all_confidence, all_eucdist, num_samples):
    model.eval()
    losses = torch.zeros(num_samples)
    entropies = torch.zeros(num_samples)
    confidences = torch.zeros(num_samples, args.num_classes)
        
    total_features = torch.zeros(num_samples, 128)
    cls_confidences = torch.zeros(num_samples)
    euc_dists = torch.zeros(num_samples)
    eval_loader = loader.run('all')
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs, features = model(inputs)
            loss = CE(outputs, targets)
            entropy = conf_penalty(outputs) * -1
            confidence = torch.softmax(outputs, dim=1)
            for b in range(inputs.size(0)):
                losses[index[b]]=loss[b]
                entropies[index[b]] = entropy[b]
                confidences[index[b]] = confidence[b]
                cls_confidences[index[b]] = confidence[b][targets[b]]
                total_features[index[b]] = features[b]
                
    all_loss.append(losses)
    all_entropy.append(entropies)
    all_confidence.append(confidences)
    
    prob = np.zeros(num_samples)
    confident_class = [0 for cls in range(args.num_classes)]
    perform_selection_class = [0 for cls in range(args.num_classes)]
    for cls in range(args.num_classes):
        cls_ind = np.where(noisy_targets==cls)[0]
        # Below process are conducted in selection_or_not function
        # 1) Recognition of Class Readiness
        # 2) Clean-Dominant Class Identification        
        perform_selection, confidence_threshold, confident = selection_or_not(cls, confidences[cls_ind])
        if confident:
            confident_class[cls]=1

        if perform_selection:
            perform_selection_class[cls] = 1
            temp_confidences = cls_confidences[cls_ind]
            temp_criteria = torch.where(temp_confidences>=confidence_threshold)[0] # representative sample set index
            
            high_index = cls_ind[temp_criteria]
            high_confidence_features = total_features[high_index]
            if len(temp_criteria)==1:
                high_confidence_features = high_confidence_features.unsqueeze(0)
            centroid = torch.mean(high_confidence_features, axis=0) # Eq.(3) calculation of prototype
            cls_features = total_features[cls_ind] # feature vector of individual samples in each class
            
            euclidean_dist = torch.norm(cls_features - centroid.unsqueeze(0), p=2, dim=1) # Euclidean distance from prototype and each sample
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
    all_eucdist.append(euc_dists)
        
    class_avg_confidence = []
    for cls in range(args.num_classes):
        class_avg_confidence.append({})
        
    for cls in range(args.num_classes):
        cls_clean_ind = np.where((noisy_targets==cls) & (clean_targets==cls))[0]
        cls_noisy_ind = np.where((noisy_targets==cls) & (clean_targets!=cls))[0]
        
        clean_confidences = torch.mean(confidences[cls_clean_ind], axis=0)
        noisy_confidences = torch.mean(confidences[cls_noisy_ind], axis=0)
        for _cls in range(args.num_classes):
            class_avg_confidence[cls][f'clean{_cls}'] = [clean_confidences[_cls].item()]
            class_avg_confidence[cls][f'noisy{_cls}'] = [noisy_confidences[_cls].item()]
        
    selected_index = (prob>0.5) # returns True for 'selected as clean samples' and 'samples in each unready class'
    selected_index = selected_index.nonzero()[0]
    selected_prob = prob[selected_index]
        
    if sum(perform_selection_class)==args.num_classes:
        bce_weight = 0
    else:
        bce_weight = 1
    
    return selected_index, selected_prob, all_loss, all_jsd, all_entropy, all_confidence, all_eucdist, confident_class, bce_weight

# evaluation on the selected samples
def selection_results(selection):
    
    selection_precision_recall = {}
    
    selection = np.array(selection)
    for i in range(args.num_classes):
        cls_noisy_inds = np.where((noisy_targets==i))[0]
        cls_selected_inds = np.intersect1d(cls_noisy_inds, selection)
        
        cls_clean_inds = np.where((noisy_targets==i) & (clean_targets==i))[0]
        cls_selected_clean_inds = np.intersect1d(cls_selected_inds, cls_clean_inds)
        
        if len(cls_selected_inds) == 0:
            cls_precision = 0
        else:
            cls_precision = len(cls_selected_clean_inds) / len(cls_selected_inds)
        cls_recall = len(cls_selected_clean_inds) / len(cls_clean_inds)
        
        selection_precision_recall[f'{i}_precision'] = [cls_precision]
        selection_precision_recall[f'{i}_recall'] = [cls_recall]
        cls_precision = round(cls_precision, 3)
        cls_recall = round(cls_recall, 3)
        print(f'Class{i} ({cls_precision}/{cls_recall})', end=' ')
    print()
    
    selection_df = pd.DataFrame(selection_precision_recall)
    
    selection_dir = save_warmup_path+'/selection_results.csv'
    if not os.path.exists(selection_dir):
        selection_df.to_csv(selection_dir, index=False, mode='w')
    else:
        selection_df.to_csv(selection_dir, index=False, mode='a', header=False)  

def main(warm_up_flag):

    all_loss = [[],[]] # save the history of losses from two networks
    all_jsd = [[],[]] # save the history of losses from two networks
    all_entropy = [[],[]]
    all_confidence = [[],[]]
    all_eucdist = [[],[]]

    if not warm_up_flag:
        net1.load_state_dict(torch.load(save_warmup_path+'%s_%.1f_%s_%.1f'%(args.dataset,args.imb_factor,args.corruption_type,args.corruption_prob)+'warm_net1.pt'))
        net2.load_state_dict(torch.load(save_warmup_path+'%s_%.1f_%s_%.1f'%(args.dataset,args.imb_factor,args.corruption_type,args.corruption_prob)+'warm_net2.pt'))
     
    for epoch in range(args.num_epochs+1):       
        
        if  warm_up_flag: 
            if  epoch<warm_up:     
                warmup_trainloader = loader.run('warmup')
                print('Warmup Net1')
                warmup_net(epoch,net1,optimizer1,warmup_trainloader)
                print('\nWarmup Net2')
                warmup_net(epoch,net2,optimizer2,warmup_trainloader) 
                if epoch == warm_up -1:
                    warm_up_flag = False
        else:
            if epoch < warm_up:
                continue 
            else:
                # construct sample set using each model
                with torch.no_grad():
                    select_index_1, select_prob_1, all_loss[0], all_jsd[0], all_entropy[0], all_confidence[0], all_eucdist[0], confident_class1, bce_weight1 = eval_train(net1, all_loss[0], all_jsd[0], all_entropy[0], all_confidence[0], all_eucdist[0], num_samples)
                    selection_results(select_index_1)
                    
                    select_index_2, select_prob_2, all_loss[1], all_jsd[1], all_entropy[1], all_confidence[1], all_eucdist[1], confident_class2, bce_weight2 = eval_train(net2, all_loss[1], all_jsd[1], all_entropy[1], all_confidence[1], all_eucdist[1], num_samples)
                    selection_results(select_index_2)

                print('Train Net1')
                labeled_trainloader, unlabeled_trainloader = loader.run('train',select_index=select_index_2,prob = select_prob_2) # co-divide
                if unlabeled_trainloader==None:
                    # If all classes are considered to be unready classes, we train model using supervised objective
                    train_supervised(epoch,net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader, confident_class2, bce_weight2) # train net1\
                else:
                    # train net1 based on the selection results from net2
                    train(epoch,net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader, confident_class2, bce_weight2) # train net1  
        
                print('\nTrain Net2')
                labeled_trainloader, unlabeled_trainloader = loader.run('train',select_index=select_index_1,prob =select_prob_1) # co-divide                
                if unlabeled_trainloader==None:
                    # If all classes are considered to be unready classes, we train model using supervised objective
                    train_supervised(epoch,net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader, confident_class1, bce_weight1) # train net2
                else:
                    # train net2 based on the selection results from net1
                    train(epoch,net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader, confident_class1, bce_weight1) # train net2  
    
        test(epoch,net1,net2) 
        
        temp_dict = {'loss':all_loss, 'clean_targets':clean_targets, 'noisy_targets':noisy_targets, 'confidence':all_confidence, 'eucdist':all_eucdist}
        pickle_dir = os.path.join(save_warmup_path, 'All_Loss.pickle')
        with open(pickle_dir,'wb') as fw:
            pickle.dump(temp_dict, fw)          
        

if __name__ == '__main__':
    main(warm_up_flag)
    
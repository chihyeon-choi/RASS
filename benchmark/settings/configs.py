#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import os
import json
import torch
import random
import argparse
import numpy as np

def parse_arguments():
	# parse arguments
	parser = argparse.ArgumentParser(description='Imbalanced Example')
	parser.add_argument('--dataset', default='cifar10', type=str,
	                    help='dataset (cifar10 [default] or cifar100)')
	parser.add_argument('--corruption_prob', type=float, default=0.4,
	                    help='label noise')
	parser.add_argument('--corruption_type', '-ctype', type=str, default='flip',
	                    help='Type of corruption ("unif" or "flip" or "flip2").')
	parser.add_argument('--batch-size', type=int, default=64, metavar='N',
	                    help='input batch size for training (default: 64)')
	parser.add_argument('--num_classes', type=int, default=10)
	parser.add_argument('--imb_factor', type=float, default=0.1)
	parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
	                    help='input batch size for testing (default: 100)')
	parser.add_argument('--epochs', type=int, default=100, metavar='N',
	                    help='number of epochs to train')
	parser.add_argument('--lr', '--learning-rate', default=0.02, type=float,
	                    help='initial learning rate')
	parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
	parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
	parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
	                    help='weight decay (default: 5e-4)')
	parser.add_argument('--no-cuda', action='store_true', default=False,
	                    help='disables CUDA training')
	parser.add_argument('--seed', type=int, default=25, metavar='S',
	                    help='random seed (default: 42)')
	parser.add_argument('--print-freq', '-p', default=50, type=int,
	                    help='print frequency (default: 10)')
	parser.add_argument('--model_name', default='baseline', type=str,
	                    help='use model version')
	parser.add_argument('--exp_name', default='', type=str,
	                    help='experiment name')
	parser.add_argument('--start_epoch', type=int, default=0, metavar='N',
	                    help='start epoch to train')
	parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
	parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
	parser.add_argument('--num_epochs', default=100, type=int)
	parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
	parser.add_argument('--alpha', default=3.0, type=float) 
	parser.add_argument('--tau', default=0.8, type=float) 
	parser.add_argument('--constant_bce', default=1.0, type=float)
	parser.add_argument('--data_path', default='./datas/data/cifar-10-batches-py', type=str, help='path to dataset')
	parser.add_argument('--penalty', default=1.0, type=float)
	args = parser.parse_args()
	print(args)
	return args


def set_seed(seed=None):
	# control randomness
	if seed:
		# random #
		random.seed(seed)  
		# hashseed #
		os.environ['PYTHONHASHSEED'] = str(seed)  
		# numpy #
		np.random.seed(seed)  
		# torch #
		torch.manual_seed(seed)  # torch cpu
		torch.cuda.manual_seed(seed)  # torch current gpu
		torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
		torch.backends.cudnn.benchmark = False  # ensure CUDA selects the same algorithm each time an application is run
		torch.backends.cudnn.deterministic = True  # Avoiding nondeterministic algorithms
		# torch.set_deterministic(True) # make other PyTorch operations behave deterministically

def set_device(args):
	use_cuda = not args.no_cuda and torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	return device
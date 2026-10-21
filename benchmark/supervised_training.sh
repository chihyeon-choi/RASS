# arguments
# dataset: 'cifar10' if CIFAR-10 / 'cifar100' if CIFAR-100
# num_class: '10' if CIFAR-10 / '100' if CIFAR-100
# Imb factor (rho): 0.1 \in {0.1, 0.05}
# corruption_type: the type of synthetic label noise \in {'unif', 'filp'} / 'unif' => symmetric, 'filp' => asymmetric
# corruption_prob: noise ratio
# alpha: confidence margin
# tau: confidence threshold


# CIFAR-10 semi-supervised objective
python ./experiment_ours_selection/RASS_benchmark_supervised.py --dataset cifar10 --num_class 10 --imb_factor 0.1 --corruption_type unif --corruption_prob 0.4 --seed 0 --alpha 3.0 --tau 0.8
python ./experiment_ours_selection/RASS_benchmark_supervised.py --dataset cifar10 --num_class 10 --imb_factor 0.1 --corruption_type unif --corruption_prob 0.6 --seed 0 --alpha 3.0 --tau 0.8
python ./experiment_ours_selection/RASS_benchmark_supervised.py --dataset cifar10 --num_class 10 --imb_factor 0.05 --corruption_type unif --corruption_prob 0.4 --seed 0 --alpha 3.0 --tau 0.8
python ./experiment_ours_selection/RASS_benchmark_supervised.py --dataset cifar10 --num_class 10 --imb_factor 0.05 --corruption_type unif --corruption_prob 0.6 --seed 0 --alpha 3.0 --tau 0.8
python ./experiment_ours_selection/RASS_benchmark_supervised.py --dataset cifar10 --num_class 10 --imb_factor 0.1 --corruption_type flip --corruption_prob 0.2 --seed 0 --alpha 3.0 --tau 0.8
python ./experiment_ours_selection/RASS_benchmark_supervised.py --dataset cifar10 --num_class 10 --imb_factor 0.1 --corruption_type flip --corruption_prob 0.4 --seed 0 --alpha 3.0 --tau 0.8
python ./experiment_ours_selection/RASS_benchmark_supervised.py --dataset cifar10 --num_class 10 --imb_factor 0.05 --corruption_type flip --corruption_prob 0.2 --seed 0 --alpha 3.0 --tau 0.8
python ./experiment_ours_selection/RASS_benchmark_supervised.py --dataset cifar10 --num_class 10 --imb_factor 0.05 --corruption_type flip --corruption_prob 0.4 --seed 0 --alpha 3.0 --tau 0.8

# CIFAR-100 semi-supervised objective
python ./experiment_ours_selection/RASS_benchmark_supervised.py --dataset cifar100 --num_class 100 --imb_factor 0.1 --corruption_type unif --corruption_prob 0.4 --seed 0 --alpha 3.0 --tau 0.8 --data_path ./datas/data/cifar-100-python
python ./experiment_ours_selection/RASS_benchmark_supervised.py --dataset cifar100 --num_class 100 --imb_factor 0.1 --corruption_type unif --corruption_prob 0.6 --seed 0 --alpha 3.0 --tau 0.8 --data_path ./datas/data/cifar-100-python
python ./experiment_ours_selection/RASS_benchmark_supervised.py --dataset cifar100 --num_class 100 --imb_factor 0.05 --corruption_type unif --corruption_prob 0.4 --seed 0 --alpha 3.0 --tau 0.8 --data_path ./datas/data/cifar-100-python
python ./experiment_ours_selection/RASS_benchmark_supervised.py --dataset cifar100 --num_class 100 --imb_factor 0.05 --corruption_type unif --corruption_prob 0.6 --seed 0 --alpha 3.0 --tau 0.8 --data_path ./datas/data/cifar-100-python
python ./experiment_ours_selection/RASS_benchmark_supervised.py --dataset cifar100 --num_class 100 --imb_factor 0.1 --corruption_type flip --corruption_prob 0.2 --seed 0 --alpha 3.0 --tau 0.8 --data_path ./datas/data/cifar-100-python
python ./experiment_ours_selection/RASS_benchmark_supervised.py --dataset cifar100 --num_class 100 --imb_factor 0.1 --corruption_type flip --corruption_prob 0.4 --seed 0 --alpha 3.0 --tau 0.8 --data_path ./datas/data/cifar-100-python
python ./experiment_ours_selection/RASS_benchmark_supervised.py --dataset cifar100 --num_class 100 --imb_factor 0.05 --corruption_type flip --corruption_prob 0.2 --seed 0 --alpha 3.0 --tau 0.8 --data_path ./datas/data/cifar-100-python
python ./experiment_ours_selection/RASS_benchmark_supervised.py --dataset cifar100 --num_class 100 --imb_factor 0.05 --corruption_type flip --corruption_prob 0.4 --seed 0 --alpha 3.0 --tau 0.8 --data_path ./datas/data/cifar-100-python
# Understanding the Role of Training Regimes in Continual Learning
Towards increasing stability of neural networks for continual learning (NeurIPS'20)

**Note: I will add an updated version of the code soon. If you have problem reproducing the results, please see the instructions for reproducing [experiment 1](https://github.com/imirzadeh/stable-continual-learning/issues/1) and [experiment 2](https://github.com/imirzadeh/stable-continual-learning/issues/5).**


## 1. Code Structure
The high level structure of the code is as follows:

```
root
├── stable_sgd
│   
└── external_libs
    └── continual_learning_algorithms
    └── hessian_eigenthings
```

1. `stable_sgd`   : implementations of our stable and plastic training regimen for SGD (in Pytorch).      
2. `external_libs`: third-party implementations we used for our experiments such as:   
    2.1 `continual_learning_algorithms` Open source implementations for A-GEM, ER-Reservoir, and EWC (in Tensorflow).   
    2.2 `hessian_eigenthings`: Open source implementation of deflated power iteration for eigenspectrum calculations (in Pytorch).  

## 2. Setup & Installation
The code is tested on Python 3.6+, PyTorch 1.5.0, and Tensorflow 1.15.2. In addition, there are some other numerical and visualization libraries that are included in ``requirements.txt`` file. However, for convenience, we provide a script for setup:   
```
bash setup_and_install.sh
```

## 3. Replicating the Results
Note: I will add an updated version of the code soon. If you have problem reproducing the results, please see the instructions for reproducing [experiment 1](https://github.com/imirzadeh/stable-continual-learning/issues/1) and [experiment 2](https://github.com/imirzadeh/stable-continual-learning/issues/5).

We provide scripts to replicate the results:   
 * 3.1 Run ```bash replicate_experiment_1.sh``` for experiment 1 (stable vs plastic).   
 * 3.2 Run ```bash replicate_experiment_2.sh``` for experiment 2 (Comparison with other methods with 20 tasks).
 * 3.3 Run ```bash replicate_appendix_c5.sh```  for the experiment in appendix C5 (Stabilizing other methods).
 
For faster replication, here we have only 3 runs per method per experiment, but we used 5 runs for the reported results.

# My Changes

## Edits
Added in NormOut layers into the models; unclear how well these will work in ResNets.

## Instructions for Running
Request a node with `srun -n 6 --mem 40G --pty -t 10:00:00 -p gpu --gres=gpu:teslaV100:1 bash`, activate your relevant environment (Xander: `source activate sdm_env`, Deepak: `conda activate env_pytorch`), then call `module load gcc/9.2.0`.
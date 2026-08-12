# Lonic: Algorithm-Hardware Co-Design for Energy-Efficient Fully Local Online SNN Training with INT4 Precision
This repository contains the code asscoiated with "Lonic: Algorithm-Hardware Co-Design for Energy-Efficient Fully
Local Online SNN Training with INT4 Precision", accepted to ICCAD2026. It contains the algorithm and simulator for Lonic.

## Introduction
Spiking neural networks (SNNs) have recently attracted increasing attention as an energy-efficient learning paradigm. Existing
works also propose temporally and fully local online SNN training algorithms to address memory and computation overhead.
However, they do not consider whether the algorithmic advantages can be effectively translated into real-device efficiency. To
address this challenge, we present Lonic, an algorithm-hardware co-design for energy-efficient and scalable fully local online supervised SNN learning. On the algorithm side, we implement an
INT4 low-precision training algorithm for fully local online SNN learning while maintaining accuracy. On the hardware side, to leverage the benefits of the proposed algorithm, we introduce reconfigurable multiplier-free integer PE arrays, dual-optimization zero-gating strategy, temporal prefix-accelerated local learning dataflow, and low-precision weight movement to significantly improve training efficiency. Compared to Apple M4 and Nvidia V100 GPUs, Lonic achieves average energy efficiency improvements of 17.44x and 66.28x, respectively, along with speedups of 3.25x and 1.02x respectively. Moreover, Lonic achieves 15.95x (14.64x) and 1.52x (7.28x) energy efficiency (area efficiency) over ASIC TPU-like and H2Learn accelerators, respectively. 

## Run Algorithm
Use the following command to get started:
```shell
python main.py --dataset CIFAR10 --arch cifar_tessvgg_model --data-path ~/Datasets --save-path ./experiments/CIFAR10_VGG_Lonic_wn_int4_ternary_int8 --trials 1 --epochs 100 --batch-size 1 --val-batch-size 64 --print-freq 5000 --delay-ls 6 --factors-stdp 0.2 0.5 1 1 --pooling MAX --scheduler 100 --lr 0.0000078125 --lr-conv 0.0000078125 --experiment-name "CIFAR10" --training-mode tess --loss "CE" --wn --optimizer Adam --low-precision-training --seed 1 --forward-intx 4
```
The output can be found in ./algorithm/experiments/.

## Run Simulator
Please look at the Jupyter Notebooks.

## Citation
to be updated

## Reference Repositories
TESS: https://github.com/mapolinario94/TESS
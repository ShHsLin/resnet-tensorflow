# ResNet Tensorflow on CIFAR10

This repository provides implementation to reproduce the result of ResNetv1 from the paper [Deep Residual Learning 
for Image Recognition](https://arxiv.org/abs/1512.03385) on CIFAR10 in
Tensorflow.

In addition, implementation of compressed resnetv1 using Tensor Train decomposition, named as resnet-v1-tt, is provided. The tt-layer is taken from [TensorNet](https://github.com/timgaripov/TensorNet-TF). Different architectures such as ResNetv2,  wide ResNet, ResNext are left as future work. For more detail, please check the reference papers.


Note all papers, the network used for ImageNet and CIFAR10 are different.
Basically, the network used for CIFAR10 has less number of channels and does
not down sample immediately after first conv layer. If one naively train the 
network for ImageNet on CIFAR10 dataset, the result can not
reach the expected value as in the paper due to overfitting.

![](Figures/cifar10.png) 

### Requirements
python2.7, tensorflow 1.0 or later version
 

### Commands

To run the train.py, one need to specify the hyperparameter in the arguments,
for example    
```
python train.py --net v1_29_tt --batch_size 64 --ckpt_dir Model/CIFAR10/v1_29_tt_b60_reg --opt SGD --regu 0.0001 --bond_dim 60 --init_step 0 --lr 1e-1
```

One also need to specific the directory for checkpoint and which network is
used during testing, for example
```
python test.py --net v1_29_tt --ckpt_dir Model/CIFAR10/v1_29_tt_reg
```


Reuse Pretrain model from
[tensorflow/models/slim](https://github.com/tensorflow/models/tree/master/slim)
One can rename the variables by the bash script, for example
```
bash rename_resnet_v1_50.sh
```
then change the pretrain in train.py as True, put the pretrain model in
corresponding folder. One can then load the pretrain model and fine tune.


### Data Augmentation
Zero padding to 40x40, random crop to 32x32, random horizontal flipping, per
image whitening ( per image standardization)


### Results

Train:45000, Val:5000   
ResNet56 (2+54) bottleneck channels = [16-32-64], reg 0.0001   
Accuracy: 91.8%  

Train:50000, Val:10000   
ResNet29 (2+27) bottleneck channels = [16-32-64], reg 0.0001   
Accuracy: 89%   
ResNet29 (2+27) bottleneck channels = [16-32-64], reg 0.0003   
Accuracy: 91.04%   
ResNet29 (2+27) bottleneck channels = [64-128-256], reg 0.0005   
Accuracy: 94.0%   # of para: 4.9M  
  
ResNet29-tt (2+27) bottleneck channels = [64-128-256], bond-dim = 30, # of  para: 0.42M    
Accuracy: 87.4%        reg 0    
Accuracy: 90.7%      reg 1e-4    
Accuracy: 90.93%      reg 5e-4    


ResNet29-tt (2+27) bottleneck channels = [64-128-256], bond-dim = 60, # of para: 1.2M   
Accuracy: 87.76%	reg 0    
Accuracy: 90.77%	reg 1e-4    
Accuracy: 89.71% 	reg 5e-4    



### To do list

- [ ] Change to resnet block take bond-dim as arguments, so one can have
  flexible bond-dim.
- [ ] Remove unnecessary VGG related code
- [ ] Separate numpy save to utils
- [ ] ResNet version2 as in arXiv:1603.05027

### Reference  
ResNetv1 50layers Visualization [here](http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006)     
ResNetv1 [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)   
ResNetv2 [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)    
Wide Residual Network  [here](https://arxiv.org/pdf/1605.07146.pdf)       
[Wider or Deeper: Revisiting the ResNet Model for Visual
Recognition](https://arxiv.org/abs/1611.10080)    
ResNeXt (ResNetv3) [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)   
[Ultimate tensorization: compressing convolutional and FC layers alike](https://arxiv.org/abs/1611.03214)     


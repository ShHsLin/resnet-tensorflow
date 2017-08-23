# ResNet Tensorflow on CIFAR10

This is an implementation of ResNet v1 in Tensorflow.
Here, the ResNet v1 refer to the network in [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385),
where different architectures such as ResNet v2  [Identity Mappings in Deep Residual
Networks](https://arxiv.org/pdf/1603.05027.pdf), wide ResNet 
[Wider or Deeper: Revisiting the ResNet Model for Visual Recognition](https://arxiv.org/abs/1611.10080) is left as future work.

Note all papers, the network used for ImageNet and CIFAR10 are different.
Basically, the network used for CIFAR10 has less number of channels. If one
naively train the network for ImageNet on CIFAR10 dataset, the result can not
reach the expected value as in the paper due to overfitting.

![](Figures/cifar10.png) 


https://github.com/KaimingHe/deep-residual-networks#disclaimer-and-known-issues
ResNet50 Visualization [here](http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006)
Wide Residual Network  [here](https://arxiv.org/pdf/1605.07146.pdf)



### To do list

- [ ] Remove unnecessary VGG related code
- [ ] Separate numpy save to utils
- [ ] ResNet version2 as in arXiv:1603.05027
- [ ] 

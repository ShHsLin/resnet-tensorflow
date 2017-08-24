## rename resnet50 from slim to my conventionn

python tensorflow_rename_variables.py --checkpoint_dir='../Model/' --replace_from='/bottleneck_v1/' --replace_to='/'
python tensorflow_rename_variables.py --checkpoint_dir='../Model/' --replace_from='resnet_v1_50/conv1/' --replace_to='resnet_v1_50/block0/conv1/'
python tensorflow_rename_variables.py --checkpoint_dir='../Model/' --replace_from='shortcut/BatchNorm/' --replace_to='shortcut/bn/'
python tensorflow_rename_variables.py --checkpoint_dir='../Model/' --replace_from='conv1/BatchNorm/' --replace_to='bn1/'
python tensorflow_rename_variables.py --checkpoint_dir='../Model/' --replace_from='conv2/BatchNorm/' --replace_to='bn2/'
python tensorflow_rename_variables.py --checkpoint_dir='../Model/' --replace_from='conv3/BatchNorm/' --replace_to='bn3/'

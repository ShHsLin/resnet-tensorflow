from network.resnet_v1 import resnet_v1_29
from network.resnet_v1 import resnet_v1_29_tt


def select_net(which_net, input_rgb, num_classes, is_training=True):
    if which_net == "v1_29":
        net = resnet_v1_29(input_rgb=input_rgb,
                           num_classes=num_classes,
                           is_training=is_training)
    elif which_net == "v1_29_tt":
        net = resnet_v1_29_tt(input_rgb=input_rgb,
                              num_classes=num_classes,
                              is_training=is_training)
    else:
        raise NotImplementedError

    return net

def select_opt(which_opt):
    return opt

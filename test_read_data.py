import read_data
params={}
params['data_path']='../CIFAR10/cifar-10-batches-py'
params['batch_size']=64
params['mode']=True
C = read_data.CIFAR10(params)

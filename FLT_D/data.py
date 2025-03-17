
import torch

import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader, Subset

# 根据数据获取数据集
def getDataset(shuju,logger):
    # cifar 10
    logger.info(f'now is {shuju} dataset')
    if shuju == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # 加载数据集
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)  
    return trainset, testset

def getSplit(trainset, testset, num_client, attack_id):
    client_datasets = []
    client_datasets_test = []
    # 默认五个客户端，trainset由6w个样本
    for i in range(num_client):
        # 每个客户端选取10000个样本执行训练
        client_datasets.append(Subset(trainset, list(range(i*10000, i*10000 + 10000))))
    client_datasets_test = Subset(testset, list(range(0, 1000)))
    
    # 目标数据集20个样本，成员与非成员各10个
    target_mem_dataset = Subset(trainset, list(range(0, 1000)))
    target_nmem_dataset = Subset(testset, list(range(0, 1000)))
    # 攻击者数据集同样配置
    attack_mem_dataset = Subset(trainset, list(range(attack_id*1000, attack_id*1000 + 1000)))
    attack_nmem_dataset = Subset(testset, list(range(attack_id*1000, attack_id*1000 + 1000)))
    
    return client_datasets, client_datasets_test, [target_mem_dataset, target_nmem_dataset], [attack_mem_dataset, attack_nmem_dataset]
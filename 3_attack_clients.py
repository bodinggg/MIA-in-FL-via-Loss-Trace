import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader, Subset, TensorDataset, ConcatDataset,random_split
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import copy
import numpy as np


import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import log_loss
from scipy.interpolate import interp1d

from model.Net import Net
from model.VGG11 import VGG11

from datetime import datetime

#当前数据
shuju = 'cifar10'
attack_round = 1
local_epochs = 10
global_label = 1
if shuju == 'cifar10':
    num_class = 10
if shuju == 'cifar100':
    num_class = 100

# 初始化全局模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('now is cuda !!!')


def ROC_AUC_Result_logshow(label_values, predict_values, reverse=False):
    if reverse:
        pos_label = 0
        print('AUC = {}'.format(1 - roc_auc_score(label_values, predict_values)))
    else:
        pos_label = 1
        print('AUC = {}'.format(roc_auc_score(label_values, predict_values)))
    fpr, tpr, thresholds = roc_curve(label_values, predict_values,
                                     pos_label=pos_label)
    print("Thresholds are {}. The len of Thresholds is {}".format(thresholds, len(thresholds)))
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic(ROC)')
    plt.loglog(fpr, tpr, 'b', label='AUC=%0.4f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0.001, 1], [0.001, 1], 'r--')
    plt.xlim([0.001, 1.0])
    plt.ylim([0.001, 1.0])
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    ax = plt.gca()
    line = ax.lines[0]
    xdata = line.get_xdata()
    ydata = line.get_ydata()
    f = interp1d(xdata, ydata)
    fpr_0 = 0.001
    tpr_0 = f(fpr_0)
    with open('result.txt','a',encoding='utf-8') as file:
        file.write(f'TPR at 0.1% FPR is {tpr_0}')
    print('TPR at 0.1% \FPR is {}'.format(tpr_0))
    plt.show()
# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])




if shuju == 'cifar10':
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # 加载CIFAR10数据集
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
if shuju == 'cifar100':
    # define transforms
    transform = transforms.Compose([
            transforms.Resize((227,227)),
            transforms.ToTensor(),
            transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    ),
    ])
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)


def label_dataloader(dataloader):
    data_list = []
    label_list = []
    cnt = 0
    for data, labels in dataloader:
        for i in range(data.shape[0]):
            data_list.append(data[i].numpy())
            label_list.append([labels[i].numpy(), cnt])
            cnt += 1
    data_list = np.array(data_list)
    label_list = np.array(label_list)

    data_list = torch.from_numpy(data_list)
    label_list = torch.from_numpy(label_list)
    tgt_dataset = TensorDataset(data_list, label_list)
    tgt_dataloader = DataLoader(tgt_dataset, batch_size=16,shuffle=True)
    return tgt_dataloader, cnt

# 划分数据集为7个部分
num_clients = 6
dataset_size = len(trainset)
dataset_size_test = len(testset)
client_datasets = []
client_datasets_test = []
for i in range(num_clients):
    start = i * dataset_size // num_clients
    end = (i + 1) * dataset_size // num_clients
    start_test = i * dataset_size_test // num_clients
    end_test = (i + 1) * dataset_size_test // num_clients

    subset_temp = Subset(trainset, list(range(start, end)))

    client_datasets.append(subset_temp)
    client_datasets_test.append(Subset(testset, list(range(start_test, end_test))))


# 创建DataLoader
client_data_loaders = [DataLoader(client_dataset, batch_size = 32, shuffle=True) for client_dataset in client_datasets]
client_data_loaders_test = [DataLoader(client_dataset_test, batch_size = 32, shuffle=True) for client_dataset_test in client_datasets_test]

########################################################
# 定义高斯模糊变换
if shuju == 'cifar100':
    gaussian_transform = transforms.Compose([
        transforms.Resize((227,227)),
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(1.0, 5.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
if shuju == 'cifar10':
    gaussian_transform = transforms.Compose([
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(1.0, 5.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# 干扰后的集合
perturbed_client_data_loader_list = []
for i in range(6):
    # 获取client_data_loaders[0]的索引
    indices = client_data_loaders[i].dataset.indices

    # 创建一个新的数据集，应用高斯模糊变换
    if shuju == 'cifar100':
        perturbed_trainset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=gaussian_transform
        )
    if shuju == 'cifar10':
        perturbed_trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=gaussian_transform
        )

    # 使用高斯模糊变换的数据集创建一个新的子集
    perturbed_client_dataset = Subset(perturbed_trainset, indices)

    # 创建一个新的DataLoader来处理扰动后的数据
    perturbed_client_data_loader = DataLoader(
        perturbed_client_dataset, batch_size = 32, shuffle=True
    )
    perturbed_client_data_loader_list.append(perturbed_client_data_loader)
print(f'perturbed_client_data_loader_list is {perturbed_client_data_loader_list}')
###########################################################


attack_loss = [] #"成员" Loss参数，获取这个成员数据训练过程的所有损失进行分类
non_mem_loss = []
target_loss = []

if shuju == 'cifar10':   
    model = Net().to(device)
    model_path = 'crt_model_50_10.pth'
if shuju == 'cifar100':
    model = VGG11().to(device)
    model_path = 'model_11_5_cifar100.pth'

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()

model = torch.load(model_path)

threshold_global = 0
global_target_list = []
def get_target_dataloader(dataloader_mem, dataloader_nmem, is_target):
    data_list = []
    label_list = []
    blur = transforms.GaussianBlur(kernel_size=7, sigma=(2.0, 2.0))
    # # 应用到图像上
    # blurred_image = blur(image)
    threshold = 0
    cnt = 0
    # 从 dataloader_mem 中收集数据和标签
    # 限制成员数据集大小更合理，看给定义样本中是否存在成员
    mem_cnt = 0
    mem_limit = len(dataloader_mem.dataset)
    if is_target:
        mem_limit = len(dataloader_mem.dataset)
    for data, label in dataloader_mem:
        for i in range(data.shape[0]):
            if mem_cnt > mem_limit:
                break  
            label_list_temp = [label[i].cpu().detach().numpy(),cnt]            
            mem_cnt += 1
            if is_target:
                global_target_list.append(label[i].item())
            data_list.append(data[i].cpu().detach().numpy())
            label_list.append(label_list_temp)
            threshold += 1
            cnt += 1

    # 从 dataloader_nmem 中收集数据和标签
    # 现在想限制一下nmem数据集大小，看目标数据集不平衡时候会怎么表现
    nmem_cnt = 0
    nmem_limit = len(dataloader_nmem.dataset)
    if is_target:
        nmem_limit = len(dataloader_nmem.dataset)
    for data, label in dataloader_nmem:
        for i in range(data.shape[0]):
            if nmem_cnt > nmem_limit:
                break
            label_list_temp = [label[i].cpu().detach().numpy(),cnt]
            nmem_cnt += 1
            if is_target:
                global_target_list.append(label[i].item())
            data_list.append(data[i].cpu().detach().numpy())
            label_list.append(label_list_temp)
            cnt += 1
    print(f'最终， cnt为 {cnt}')
    data_list = np.array(data_list)
    label_list = np.array(label_list)

    data_list = torch.from_numpy(data_list)
    label_list = torch.from_numpy(label_list)
    print(f'[target] label_list is {label_list}')

    tgt_dataset = TensorDataset(data_list, label_list)

    tgt_dataloader = DataLoader(tgt_dataset, batch_size = 16,shuffle=True)

    return tgt_dataloader,threshold, cnt, mem_cnt, nmem_cnt

def get_dataloader(dataloader):
    data_list = []
    label_list = []
    dataloader_size = 0
    
    for data, target in dataloader:
        for i in range(data.shape[0]):
            data_list.append(data[i].cpu().detach().numpy())
            label_list.append(target[i])
            dataloader_size += 1
    data_list = np.array(data_list)
    label_list = np.array(label_list)

    data_list = torch.from_numpy(data_list)
    label_list = torch.from_numpy(label_list)
    print(f'[mem/non-mem] label_list is {label_list}')

    tgt_dataset = TensorDataset(data_list, label_list)

    tgt_dataloader = DataLoader(tgt_dataset, batch_size = 32,shuffle=True)

    return tgt_dataloader, dataloader_size        
            
# 想要复原一下，看看是不是get_target_dataloader写错了
def ref_target_dataloader(dataloader):
    data_list = []
    label_list = []
    temp = 0
    # 从 dataloader_mem 中收集数据和标签
    for data, label in dataloader:
        for i in range(data.shape[0]):
            temp += 1
            if temp > threshold_global:
                continue
            data_list.append(data[i].numpy())
            label_list.append(label[i].numpy())

    data_list = np.array(data_list)
    label_list = np.array(label_list)
    data_list = torch.from_numpy(data_list)
    label_list = torch.from_numpy(label_list)

    tgt_dataset = TensorDataset(data_list, label_list)

    tgt_dataloader = DataLoader(tgt_dataset, batch_size=32,shuffle=False)
    return tgt_dataloader


def Mentr(probs, true_labels):
    small_value = 1e-30
    log_probs = -np.log(np.maximum(probs, small_value))
    reverse_probs = 1 - probs
    log_reverse_probs = -np.log(np.maximum(reverse_probs, small_value))
    modified_probs = np.copy(probs)
    modified_probs[range(true_labels.size),true_labels] = reverse_probs[range(true_labels.size), true_labels]
    modified_log_probs = np.copy(log_reverse_probs)
    modified_log_probs[range(true_labels.size), true_labels] = log_probs[range(true_labels.size), true_labels]
    return np.sum(np.multiply(modified_probs, modified_log_probs),axis=1)

def calculate_entropy(probabilities):
    # 防止对0取对数，添加一个小常数epsilon
    epsilon = 1e-9
    # 计算熵值
    entropy = -torch.sum(probabilities * torch.log(probabilities + epsilon))
    return entropy

global_m_nm_dataloader, threshold_1, cnt_1, mem_cnt_1, nmem_cnt_1 = get_target_dataloader(client_data_loaders[0],client_data_loaders[4],False)
global_target_dataloader, threshold_2, cnt_2, mem_cnt_2, nmem_cnt_2= get_target_dataloader(client_data_loaders[1],client_data_loaders[5],True)
threshold_global = threshold_2



global_target_list_test = [[] for _ in range(cnt_2)]
# 训练函数
def train(client_id, model, data_loader, criterion, crt_epochs,epochs, optimizer, flag):
    print(f'哥们开始训练了，当前client_id为{client_id}，dataloader大小为{len(data_loader.dataset)}')
    model.train()
    accuracy = 0
    for epoch in range(epochs):
        # 需要跟进每一个样本去探索
        temp_attack_loss = [_ for _ in range(len(data_loader.dataset))]
        temp_non_mem_loss = [_ for _ in range(len(data_loader.dataset))]
        temp_target_loss = [_ for _ in range(len(data_loader.dataset))]
        print(f'temp_target_loss len is {len(temp_target_loss)}')
        crt = 0
        smp = 0
        epoch_loss = 0
        for data, target in data_loader:
            loss_temp = 0
            data, target = data.to(device), torch.tensor(target).to(device)
            
            optimizer.zero_grad()
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            target_index = _
            target = torch.transpose(target, 0, 1)
            target_index = target[1]
            target = target[0]
            loss = criterion(output, target)
            output_temp = output.cpu().detach().numpy()
            target_temp = target.cpu().detach().numpy()
            l_m = Mentr(output_temp,target_temp)
            for i in range(output.shape[0]):
                l_1 = criterion(output[i], target[i])
                l_2 = torch.softmax(output[i], dim = 0)[target[i]]
                l_3 = calculate_entropy(torch.softmax(output[i], dim = 0))
                if flag ==1 :
                    temp_attack_loss[target_index[i].item()] = torch.tensor(l_1)
                if flag == 2:
                    temp_non_mem_loss[target_index[i].item()] = torch.tensor(l_1)
                if flag == 3:
                    temp_target_loss[target_index[i].item()] = torch.tensor(l_1)
            loss.backward()
            optimizer.step()
            loss_temp += loss.item()
            epoch_loss += loss.item()
            crt += (predicted == target).sum().item()
            smp += target.size(0)
        if flag == 1:
            temp_attack_loss = torch.stack(temp_attack_loss)
            attack_loss.append(temp_attack_loss)
        if flag == 2:
            temp_non_mem_loss = torch.stack(temp_non_mem_loss)
            non_mem_loss.append(temp_non_mem_loss)
        if flag == 3:
            temp_target_loss = torch.stack(temp_target_loss)
            target_loss.append(temp_target_loss)
        if epoch == epochs - 1:
            accuracy = crt / smp

        print('client id :', client_id ,'current round:',crt_epochs+1,'epoch:', epoch + 1,' loss:',loss_temp ,' accuracy:', crt / smp, '\n')   
    return accuracy



# 联邦学习算法 - 使用模型平均作为聚合策略
def fedavg(client_data_loaders, global_model, round, local_epochs):
    print(f'当前数据集：{shuju}')
    client_accuracy = 0.0
    global_weights = global_model.state_dict()  #每个轮次开始时加载上次更新的全局模型参数
    temp_weights = copy.deepcopy(global_weights)
    
    attack_data_loader,_ = label_dataloader(client_data_loaders[0])
    non_mem_data_loader,_ = label_dataloader(client_data_loaders[4])

    # 获取target_data_loader
    target_data_loader = global_target_dataloader


    flag = 0    # 判断当前客户端行为
    if shuju == 'cifar10':
        client_model = Net().to(device)
    if shuju == 'cifar100':
        client_model = VGG11().to(device)
    client_model.load_state_dict(global_weights)    #加载全局模型参数

    # 定义客户端的优化器
    client_optimizer = optim.SGD(client_model.parameters(), lr=0.001, momentum=0.9)
    flag = 1
    print('!!! 客户端 A 投入 “成员” 与 “非成员” 数据集进行训练 !!!\n')
    client_accuracy = train(flag, client_model, attack_data_loader, criterion, round,local_epochs, client_optimizer, flag)
    flag = 2
    client_model.load_state_dict(global_weights)    #加载全局模型参数
    client_optimizer = optim.SGD(client_model.parameters(), lr=0.001, momentum=0.9)
    print('!!! 客户端 B 投入“目标” 数据集进行训练，获得模型参数 !!!\n')
    #第一阶段结束，接下来直接对客户端C投入“目标”数据集，进行训练，获得模型参数。
    client_accuracy = train(flag, client_model, non_mem_data_loader, criterion, round,local_epochs, client_optimizer, flag)

    flag = 3
    client_model.load_state_dict(global_weights)    #加载全局模型参数
    client_optimizer = optim.SGD(client_model.parameters(), lr=0.001, momentum=0.9)
    print('!!! 客户端 C 投入“目标” 数据集进行训练，获得模型参数 !!!\n')
    #第一阶段结束，接下来直接对客户端C投入“目标”数据集，进行训练，获得模型参数。
    client_accuracy = train(flag, client_model, target_data_loader, criterion, round,local_epochs, client_optimizer, flag)

    global_weights = temp_weights
    return client_accuracy / 2 

#获取各个数据的loss轨迹
for i in range(attack_round):
    cnt =fedavg(client_data_loaders, model, i, local_epochs=local_epochs)


print(f'global_target_list shape is {torch.tensor(global_target_list).shape}')
print(f'global_target_list_test shape is {torch.tensor(global_target_list_test).shape}')




def deal_loss(loss_data):
    loss_data = torch.stack(loss_data)
    loss_data = torch.transpose(loss_data, 0, 1)
    return loss_data

attack_loss = deal_loss(attack_loss)
label_attack_mem = torch.ones(attack_loss.shape[0], 1)

non_mem_loss = deal_loss(non_mem_loss)
label_attack_nmem = torch.zeros(non_mem_loss.shape[0], 1)

data_attack = torch.cat((attack_loss, non_mem_loss), dim = 0)
label_attack = torch.cat((label_attack_mem, label_attack_nmem), dim=0)
print(f'data_attack shape is {data_attack.shape}')
print(f'label_attack shape is {label_attack.shape};label_attack[0] is {label_attack[0]};label_attack[{attack_loss.shape[0]+1} is {label_attack[attack_loss.shape[0]+1]}]')

# 保存训练攻击模型的数据及标签
np.save(f'./test/data_attack_{shuju}.npy', data_attack.detach().cpu().numpy())
np.save(f'./test/label_attack_{shuju}.npy', label_attack.detach().cpu().numpy())

target_loss = deal_loss(target_loss)

label_target_mem = torch.ones(mem_cnt_2, 1)
label_target_nmem = torch.zeros(nmem_cnt_2 , 1)

data_target = target_loss
label_target = torch.cat((label_target_mem, label_target_nmem), dim = 0)

# 保存训练攻击模型的数据及标签
np.save(f'./test/data_target_{shuju}.npy', data_target.detach().cpu().numpy())
np.save(f'./test/label_target_{shuju}.npy', label_target.detach().cpu().numpy())

############################################################################################################

#读取已经保存的 dataloader 的 npy 文件
data_array = np.load(f'./test/data_attack_{shuju}.npy')
label_array = np.load(f'./test/label_attack_{shuju}.npy')

target_array = np.load(f'./test/data_target_{shuju}.npy')
target_labels_array = np.load(f'./test/label_target_{shuju}.npy')

print(f'target_labels mean is {np.mean(target_labels_array)}')

#将 numpy 数组转换为 torch 张量
data_tensor = torch.from_numpy(np.array(data_array))
labels_tensor = torch.from_numpy(np.array(label_array))

target_tensor = torch.from_numpy(np.array(target_array))
target_labels_tensor = torch.from_numpy(np.array(target_labels_array))

print('!!! numpy 数组转换为 torch 张量 !!!')
#创建 TensorDataset 
dataset = TensorDataset(data_tensor, labels_tensor)
dataset_target = TensorDataset(target_tensor,target_labels_tensor)
print(f'target_data shape 他妈的是 {target_tensor.shape}')

print('!!! 创建 TensorDataset !!!')
#创建 DataLoader
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle= True)
dataloader_target = DataLoader(dataset_target, batch_size=batch_size,shuffle=True)
print('!!! 创建 DataLoader !!!')



class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(local_epochs * attack_round, 1)

    def forward(self, x):
        x = torch.sigmoid(self.layer1(x))
        return x 

device = torch.device('cpu')
model_simple = SimpleModel().to(device)
model_simple.train()
#选择损失函数和优化器
criterion_simple = nn.BCELoss()
optimizer_simple = optim.Adam(model_simple.parameters(), lr=0.001)


def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"Gradient of {name}: {param.grad}")

num_epochs = 20

for epoch in range(num_epochs):
    crt = 0
    smp = 0
    for inputs,targets in dataloader:
        #将数据移动到模型所在的设备
        inputs, targets = inputs.to(device), targets.to(device)
        
        #前向传播
        outputs = model_simple(inputs)
        
        smp += targets.size(0)
        for i in range(inputs.shape[0]):
            if targets[i] == 1 and outputs[i] > 0.5:
                crt += 1
            if targets[i] == 0 and outputs[i] < 0.5:
                crt += 1
        
        loss_1 = criterion_simple(outputs, targets)   #计算损失

        optimizer_simple.zero_grad()   #清空梯度
        loss_1.backward()  # 反向传播

        optimizer_simple.step()    #更新权重
    # 测试攻击模型

    print(f'经过我的可靠的测试，这个b 模型的训练精度为{crt / smp * 100}% !!!')

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss_1.item(): .4f}')



crt_cnt = 0
smp = 0

# 收集真实标签和预测概率值
label_values = []
predict_values = []

tp = 0
fp = 0
tn = 0
fn = 0

for inputs,targets in dataloader_target:
    #将数据移动到模型所在的设备
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model_simple(inputs.float())

        probabilities = torch.sigmoid(outputs)  # 假设是二分类问题，使用sigmoid函数获取概率
        label_values.extend(targets.detach().cpu().numpy())
        predict_values.extend(probabilities.detach().cpu().numpy())

        for i in range(inputs.shape[0]):
            smp += 1
            if outputs[i] > 0.5:
                crt_cnt += 1
                if targets[i] == 1:
                    tp += 1
                if targets[i] == 0:
                    fp += 1
            if outputs[i] < 0.5:
                crt_cnt += 1
                if targets[i] == 1:
                    fn += 1
                if targets[i] == 0:
                    tn += 1
        
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall)/(precision + recall)
# with open('result.txt','a',encoding='utf-8') as file:
#     time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     file.write(f'\n{time}\n********************{shuju}*********************\n')
#     file.write(f'当前训练轮次为 {local_epochs * attack_round}, 攻击轮次为 {attack_round}\n')
#     file.write(f'attack_data_loader = client_data_loaders[3]   #“成员”数据集\n')
#     file.write(f'non_mem_data_loader = perturbed_client_data_loader_list[3]  #"非成员"数据集\n')
#     file.write(f'蒙对的几率为{((tp +tn )/smp) * 100 }%!!!恐怖如斯！！！\n')
#     file.write(f'f1得分为{f1}\n')
#     file.write(f'precision is {tp/(tp + fp)}\n')
#     file.write(f'recall is {tp/(tp + fn)}\n')
print(f'threshold_2 is {threshold_2}')
print(f'当前数据集为{shuju}')
print(f'local epochs is {local_epochs}')
print(f'蒙对的几率为{((tp + tn )/smp) * 100 }%!!!恐怖如斯！！！')
print(f'mem_cnt is {mem_cnt_2} nmem_cnt is {nmem_cnt_2}')
if mem_cnt_2 != 0:
    print(f'mem accuarcy is {tp/mem_cnt_2}')
if nmem_cnt_2 != 0:
    print(f'nmem accuarcy is {tn/nmem_cnt_2}')
print(f'f1得分为{f1}')
print(f'precision is {tp/(tp + fp)}')
print(f'recall is {tp/(tp + fn)}')
ROC_AUC_Result_logshow(label_values, predict_values)

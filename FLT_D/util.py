
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader,Subset,ConcatDataset
import torch.nn as nn
import torch.optim as optim
import copy 
import torch.nn.functional as F
import logging

from model.getModel import getModel
from model.attack_model import SimpleModel
from torch.utils.data import TensorDataset

# 定义设备，优化训练
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义日志
import logging

# 配置日志
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    # 输出到文件
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # 输出到控制台
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

# 获取可以跟踪特定样本的dataloader
def label_dataloader(data_set, batch_size):
    data_list = []
    label_list = []
    cnt = 0
    dataloader = DataLoader(data_set, batch_size = batch_size, shuffle = False)
    for data, labels in dataloader:
        for i in range(data.shape[0]):
            data_list.append(data[i].numpy())
            label_list.append([labels[i].numpy(), cnt])
            cnt += 1
    data_list = torch.from_numpy(np.array(data_list))
    label_list = torch.from_numpy(np.array(label_list))

    tgt_data_set = TensorDataset(data_list, label_list)
    tgt_dataloader = DataLoader(tgt_data_set, batch_size=batch_size, shuffle=True)
    return tgt_dataloader

# 成员与非成员混合的dataloader
def dataloader(mem_data_set, nmem_data_set, batch_size):
    data_list = []
    label_list = []
    # 成员在前
    cnt = 0
    dataloader = DataLoader(mem_data_set, batch_size = batch_size, shuffle = False)
    for data, labels in dataloader:
        for i in range(data.shape[0]):
            data_list.append(data[i].numpy())
            label_list.append([labels[i].numpy(), cnt])
            cnt += 1
    # 非成员在后
    dataloader = DataLoader(nmem_data_set, batch_size = batch_size, shuffle = False)
    for data, labels in dataloader:
        for i in range(data.shape[0]):
            data_list.append(data[i].numpy())
            label_list.append([labels[i].numpy(), cnt])
            cnt += 1
    data_list = torch.from_numpy(np.array(data_list))
    label_list = torch.from_numpy(np.array(label_list))

    tgt_data_set = TensorDataset(data_list, label_list)
    tgt_dataloader = DataLoader(tgt_data_set, batch_size=1, shuffle=False)
    return tgt_dataloader

def deal_loss_trace(loss_trace):
    loss_trace = loss_trace.view(-1, loss_trace.shape[2])
    loss_trace = torch.transpose(loss_trace, 0, 1)
    return loss_trace




# 训练和测试
# 训练函数
def train(client_id, dataloader, epochs, model, phase, logger, rnd):
    """
    client_id: 当前客户端编号
    dataloader: 当前客户端所用训练数据
    epochs：本地训练轮次
    model：当前客户端所用模型，每次开始都是获得的全局模型
    phase：当前所处全局轮次
    """
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model_list =[] 
    for epoch in range(epochs):
        torch.cuda.empty_cache()
        crt = 0
        smp = 0
        epoch_loss = 0
        cnt = 0
        temp_loss_trace = [[] for i in range(len(dataloader.dataset))]
        for data, label in dataloader:
            data, label = data.to(device), label.to(device)

            optimizer.zero_grad()
            output= model(data)
            _, predicted = torch.max(output.data, 1)
            label = torch.transpose(label, 0, 1)
            index = label[1]    # 为了跟踪单个具体数据，label与index是捆绑一起的———一一对应
            label = label[0]

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            cnt += 1
            crt += (predicted == label).sum().item()
            smp += label.size(0)
        model_list.append(copy.deepcopy(model))
        logger.info(f'rnd: {rnd+1}, client_id: {client_id},  phase: {phase},  epoch: {epoch+1},  accuracy: {crt/smp*100} %,  loss: {epoch_loss/cnt}')
    return model, model_list    # 返回最终模型和训练过程中的模型列表

def attack_train(attack_model, atk_loss_trace, attack_epochs, logger, rnd):
    # 攻击模型应执行二分类任务，根据损失轨迹维度首先创建dataloader，之后执行训练，最后返回attack_model
    criterion_attack = nn.CrossEntropyLoss()
    optimizer_attack = optim.Adam(attack_model.parameters(), lr=0.001)
    # 创建atk_loss_trace的dataloader，其中损失轨迹前1k个为成员，后1k个为非成员
    label_mem = torch.ones(1000)
    label_nmem = torch.zeros(1000)
    label = torch.concat([label_mem, label_nmem])
    label = label.unsqueeze(1)
    # 创建dataset
    data_set = TensorDataset(atk_loss_trace, label)
    # 创建对应dataloader
    data_loader = DataLoader(data_set, batch_size=1, shuffle=True)

    cmp = 2000 # 已知成员与非成员各1k
    # 训练攻击模型
    logger.info(f'now is training attack_model')
    for epoch in range(attack_epochs):
        cnt = 0
        crt = 0
        loss_sum = 0
        for data, label in data_loader:
            data, label = data.to(device), label.to(device)
            optimizer_attack.zero_grad()
            output= attack_model(data)

            for i in range(label.shape[0]):
                if label[i] == 1 and output[i] > 0.5:
                    crt += 1
                if label[i] == 0 and output[i] < 0.5:
                    crt += 1
            loss = criterion_attack(output, label)
            loss_sum += loss.item()
            cnt += 1
            loss.backward()  # 反向传播
            optimizer_attack.step()    #更新权重
        logger.info('rnd:{}, epoch: {}, loss: {}, accuracy: {}'.format(rnd+1, epoch+1,loss_sum/cnt,crt/cmp))
    return attack_model

def test(dataloader, model, logger, rnd):
    model.eval()
    crt = 0
    smp = 0
    cnt = 0
    for data, label in dataloader:
        data, label = data.to(device), label.to(device)
        label = torch.transpose(label, 0, 1)
        index = label[1]    # 为了跟踪单个具体数据，label与index是捆绑一起的———一一对应
        label = label[0]
        output= model(data)
        _, predicted = torch.max(output.data, 1)

        cnt += 1
        crt += (predicted == label).sum().item()
        smp += label.size(0)
    logger.info(f'rnd:{rnd+1}, test_accuracy: {crt/smp*100} %')
    
    return crt/smp  # 返回测试精度


# 联邦学习聚合FedAvg
def fedavg_aggregation(local_models):
    """
    :param local_models: list of local models
    :return: update_global_model: global model after fedavg
    """
    global_model = copy.deepcopy(local_models[0])
    avg_state_dict = global_model.state_dict()

    local_state_dicts = list()
    for model in local_models:
        local_state_dicts.append(model.state_dict())

    for layer in avg_state_dict.keys():
        avg_state_dict[layer] *= 0
        for client_idx in range(len(local_models)):
            avg_state_dict[layer] += local_state_dicts[client_idx][layer]
        avg_state_dict[layer] /= len(local_models)

    global_model.load_state_dict(avg_state_dict)
    return global_model

# 联邦学习算法 - 使用模型平均作为聚合策略
def fedavg(client_data_loaders, global_model, round, local_epochs, model_name, logger, rnd):

    global_weights = global_model.state_dict()  #每个轮次开始时加载上次更新的全局模型参数
    model_list = []
    for client_id, client_data_loader in enumerate(client_data_loaders):
        client_model = getModel(model_name).to(device)
        client_model.load_state_dict(global_weights)    #加载全局模型参数

        # 本地客户端执行训练，获得最终模型和模型列表
        client_model, temp_model_list= train(client_id=client_id, dataloader=client_data_loader, epochs=local_epochs, model=client_model, phase=round, logger=logger, rnd=rnd)
        if client_id == 4:
            ret_model_list = copy.deepcopy(temp_model_list)
        model_list.append(copy.deepcopy(client_model))

    # 聚合模型参数
    global_model = fedavg_aggregation(model_list)
    return global_model, ret_model_list

def get_model_list(global_model, dataloader, local_epochs, logger, rnd):
    # 获取对应数据集训练的多个模型快照
    # 10086，随便给的编号，说明现在执行的是获取用于执行攻击的模型快照
    _,model_list = train(10086, dataloader=dataloader, model=global_model,epochs= local_epochs, phase = -1,logger=logger, rnd=rnd)
    # 返回一系列攻击快照
    return model_list

# 获取MIA相关信息
def L_Trace_A(dataloader, client_model_list, distillation_model_name, distillation_epochs, logger, rnd):
    # 目标数据集与攻击者数据集需要分开获取损失轨迹，其中攻击者数据集知道成员与非成员。
    # 首先针对成员目标数据集获得损失轨迹
    mem_loss_list = []
    nmem_loss_list = []
    # 定义学生模型
    
    for i in range(len(client_model_list)):
        student_model = getModel(distillation_model_name).to(device=device)
        # 对每个模型快照执行蒸馏评估损失
        mem_loss_list_temp, nmem_loss_list_temp = distillation(client_model_list[i], student_model=student_model, epochs = distillation_epochs, data_loader= dataloader, logger=logger, rnd=rnd, local_epoch=i)
        mem_loss_list.append(mem_loss_list_temp)
        nmem_loss_list.append(nmem_loss_list_temp)

    
    # list to Tensor
    mem_loss_list = torch.stack(mem_loss_list)
    nmem_loss_list = torch.stack(nmem_loss_list)

    # 返回成员与非成员的损失轨迹
    return mem_loss_list, nmem_loss_list

def MIA(attack_model, tgt_loss_trace, logger):
    """
    attack_model: Simple_model
    tgt_loss_trace: [local_epochs*distillation_epochs, 2000]
    """
    # 正式执行MIA，通过攻击模型和目标损失轨迹，已知目标损失轨迹中前1000个为成员，后1k个为非成员
    attack_model.eval()
    # 创建tgt_loss_trace 的dataloader
    label_mem = torch.ones(1000)
    label_nmem = torch.zeros(1000)
    label = torch.concat([label_mem, label_nmem])
    label = label.unsqueeze(1)
    # 创建dataset
    data_set = TensorDataset(tgt_loss_trace, label)
    # 创建对应dataloader
    data_loader = DataLoader(data_set, batch_size=1, shuffle=True)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    smp = 0
    for data, label in data_loader:
        data, label = data.to(device), label.to(device)

        output = attack_model(data)
        for i in range(data.shape[0]):
            smp += 1
            print(f'output is {output[i]}')
            if output[i] > 0.5:
                if label[i] == 1:
                    tp += 1
                if label[i] == 0:
                    fn += 1
            if output[i] < 0.5:
                if label[i] == 1:
                    fp += 1
                if label[i] == 0:
                    tn += 1

    return tp, fp, tn, fn

# 定义蒸馏损失函数

class DistillationLoss(nn.Module):
    def __init__(self, alpha = 0, temperature = 1):
        super().__init__()
        self.alpha = alpha
        self.T = temperature
    
    def forward(self, student_outputs, teacher_outputs, labels):
        hard_loss = F.cross_entropy(student_outputs, labels)
        soft_loss = []
        for i in range(student_outputs.shape[0]):
            soft_loss.append(F.kl_div(
                F.log_softmax(student_outputs / self.T, dim=1),
                F.softmax(teacher_outputs / self.T, dim=1),
                reduction = 'batchmean'
            ) * (self.T * self.T))
        temp_soft_loss = 0
        for i in range(len(soft_loss)):
            temp_soft_loss += soft_loss[i]
        soft_loss = temp_soft_loss / len(soft_loss)
        loss = self.alpha* hard_loss + (1 - self.alpha) * soft_loss
        return loss

# 用于评估损失
def get_loss_trace(model, data_loader, logger):
    temp_loss_trace = [[] for i in range(len(data_loader.dataset))]

    for data, label in data_loader:
        data, label = data.to(device), label.to(device)
        output = model(data)

        label = torch.transpose(label, 0,1)
        index = label[1]
        label = label[0]
        for i in range(data.shape[0]): 
            criterion = nn.CrossEntropyLoss()
            l = criterion(output[i], label[i])
            # 跟踪单个样本损失轨迹
            temp_loss_trace[index[i]] = l
    # 返回loss_trace，前1k为成员
    return torch.tensor(temp_loss_trace)

# 用于评估损失
def get_confidence_trace(model, data_loader, logger):
    temp_confidence_trace = [[] for i in range(len(data_loader[0].dataset)+len(data_loader[1].dataset))]

    softmax = nn.Softmax(dim=1)
    for data, label in data_loader:
        data, label = data.to(device), label.to(device)
        output = model(data)
        output_softmax = softmax(output)
        label = torch.transpose(label, 0,1)
        index = label[1]
        label = label[0]
        for i in range(data.shape[0]): 
            l = output_softmax[i]
            # 跟踪单个样本损失轨迹
            temp_confidence_trace[index[i]] = l
    # 取均值
    mem_loss_trace = 0
    nmem_loss_trace = 0
    for i in range(5):
        mem_loss_trace += temp_confidence_trace[i].item()
        nmem_loss_trace += temp_confidence_trace[i+5].item()
    print(f'mem loss_trace:{mem_loss_trace/5}, nmem loss_trace:{nmem_loss_trace/5}')
    # 返回loss_trace，前1k为成员
    return torch.tensor(temp_confidence_trace)

# 模型蒸馏
def distillation(teacher_model, student_model, epochs, data_loader, logger, rnd, local_epoch): 
    # 定义模型蒸馏过程损失
    distillation_loss = DistillationLoss(alpha=0, temperature=1.0)  
    optimizer = optim.Adam(student_model.parameters(), lr=0.001)   
    mem_loss_trace_list = []
    nmem_loss_trace_list = []
    for epoch in range(epochs):
        crt = 0
        smp = 0
        loss_temp = 0
        cnt = 0
        # 成员执行训练与蒸馏
        for batch, label in data_loader[0]:
            batch, label = batch.to(device), label.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_outputs= teacher_model(batch)
            student_outputs = student_model(batch)
            _, predicted = torch.max(student_outputs.data, 1)
            label = torch.transpose(label, 0,1)
            index = label[1]
            label = label[0]
            loss = distillation_loss(student_outputs, teacher_outputs, label)
            loss.backward()
            optimizer.step()
            loss_temp += loss.item()
            crt += (predicted == label).sum().item()
            smp += label.size(0)
            cnt += 1
        # 当前学生模型获取每个样本损失
        mem_loss_trace_list.append(get_loss_trace(student_model, data_loader[0], logger=logger))
        nmem_loss_trace_list.append(get_loss_trace(student_model, data_loader[1], logger=logger))
        logger.info('crt rnd:{}, local_epoch:{}, epoch:{}, loss: {} ,accuracy:{}'.format(rnd, local_epoch,epoch + 1, loss_temp / cnt, crt / smp))  
    mem_loss_trace_list.append(get_loss_trace(teacher_model, data_loader[0], logger=logger))
    nmem_loss_trace_list.append(get_loss_trace(teacher_model, data_loader[1], logger=logger))
 
    logger.info(f'loss_trace len is [{len(mem_loss_trace_list)}] ')
    # 获取了成员与非成员蒸馏后评估的损失轨迹
    return torch.stack(mem_loss_trace_list),torch.stack(nmem_loss_trace_list)



def get_train_loss_trace_one(model, data, label, local_epochs):
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    loss_trace = []
    is_mem = 0
    label = torch.transpose(label, 0, 1)
    index = label[1]    # 为了跟踪单个具体数据，label与index是捆绑一起的———一一对应
    label = label[0]
    for epoch in range(local_epochs):
        optimizer.zero_grad()
        output= model(data)
        if index < 500:
            is_mem = 1
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        loss_trace.append(loss)
    loss_trace = torch.stack(loss_trace)
    return loss_trace, is_mem

"""
 这部分写一下主函数逻辑

 参与FL训练的客户端数量为N
 全局轮次为10
 本地训练轮次统一为10
 模型调用./model/下的模型
 FL聚合方式采用FedAvg

 1.选取一个客户端当作攻击者客户端，攻击者选取同分布数据集，各个客户端数据没有重叠。客户端正常获取全局模型并执行训练
    此部分函数应涉及：
    FedAvg(client_model=各个客户端模型, client_dataloaders=各个客户端数据集, globl_model=全局模型, round=当前轮次, local_epochs=本地训练轮次)
        按照顺序调用客户端，其中攻击者客户端由编号指出，全局定义。
            当前客户端执行训练过程，调用函数train(client_id=当前客户端编号, dataloader=当前数据集, local_epochs=局部轮次, model=当前模型, round=当前轮次)|return model=训练好的模型
            将当前客户端模型并入列表中，为执行聚合函数
        根据模型列表，执行模型聚合，调用函数 FedAvg(model_list=模型列表)获取模型
 2.本地客户端执行蒸馏并评估损失
    定义蒸馏函数：
    Distillation(tearcher_model=目标模型, student_model=训练学生模型, epochs=蒸馏过程轮次, dataloader=选取数据集, round=当前轮次, model_epoch=当前局部轮次)
        定义蒸馏损失（由util.py文件定义）
        执行蒸馏训练，并在蒸馏过程评估损失
        return 损失轨迹
    攻击特征获取：
    L_Trace_A(client_dataloaders=攻击者与目标数据集[成员+非成员], model=目标模型, student_model=学生模型,  local_epochs=客户端本地训练过程)
        攻击者数据集：dataloader_A=成员+非成员数据集[对于本地攻击者]
        目标数据集：dataloader_T=成员+非成员数据集[不知道具体情况不做区分考虑攻击者与成员]
        调用蒸馏函数分别获取：
        攻击者损失轨迹[知道成员与非成员对应损失轨迹的标签]
        目标损失轨迹[不知道成员与非成员对应损失轨迹的标签]
        return 攻击者和目标损失轨迹
 3.通过获取损失轨迹训练攻击模型，并对目标损失轨迹执行MIA。
    定义攻击模型，执行二分类任务（是否为成员）
    根据攻击者损失轨迹训练攻击模型
    根据目标损失轨迹执行推理攻击
"""

import torch
import torch.nn as nn
from data  import getDataset, getSplit
from util import label_dataloader, dataloader, fedavg, L_Trace_A, get_model_list, attack_train, MIA, deal_loss_trace, get_logger,test
import  copy
import numpy as np

from model.getModel import getModel
from model.attack_model import SimpleModel, SmallCNN

shuju = 'cifar10'
model_name = 'Net'
distillation_model_name = 'Net' # 考虑如果未知模型架构的实际效果会不会好
global_epochs = 20
local_epochs = 50
distillation_epochs =  0
attack_epochs = 20
num_clients = 5
attack_id = 4 # 第五个

batch_size = 32
# 定义设备，优化训练
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 使用日志
logger = get_logger(f'{shuju}_{model_name}_MIA.log')
logger.info(f'we choose {model_name} to train the target MODEL')
# 获取数据集
train_set, test_set = getDataset(shuju, logger) 

# 数据集根据客户端数量划分
client_datasets, client_datasets_test, target_datasets, attack_datasets = getSplit(train_set, test_set, num_clients, attack_id)

# 为了跟踪标签，重新设计了dataloader, 定义为label_dataloader
client_dataloaders = [label_dataloader(client_datasets[i], batch_size=batch_size) for i in range(num_clients)]
client_dataloaders_test = [label_dataloader(client_datasets_test, batch_size=batch_size) for i in range(num_clients)]

tgt_dataloader = dataloader(mem_data_set=target_datasets[0], nmem_data_set=target_datasets[1], batch_size=batch_size)
atk_dataloader = dataloader(mem_data_set=attack_datasets[0], nmem_data_set=attack_datasets[1], batch_size=batch_size)

tgt_dataloader_list = [label_dataloader(target_datasets[0],batch_size=batch_size), label_dataloader(target_datasets[1],batch_size=batch_size)]
atk_dataloader_list = [label_dataloader(attack_datasets[0],batch_size=batch_size), label_dataloader(attack_datasets[1],batch_size=batch_size)]

# 初始化全局模型
global_model = getModel(model_name=model_name).to(device=device)

def deal_loss_trace(loss_trace):
    loss_trace = torch.transpose(loss_trace, 0 ,2)
    loss_trace = torch.transpose(loss_trace, 1 ,2)
    return loss_trace

for rnd in range(global_epochs):
    # attack_client_model_list 是当前轮次下攻击者客户端的多个模型快照
    global_model, model_list = fedavg(client_data_loaders=client_dataloaders, global_model=global_model, round=rnd, local_epochs=local_epochs, model_name=model_name, logger = logger, rnd=rnd)
    # 检查当前全局模型准确率
    test(client_dataloaders_test[0], global_model, logger, rnd=rnd)

    # 获取tgt_dataloader, atk_dataloader的模型快照
    tgt_global_model = copy.deepcopy(global_model)
    atk_global_model = copy.deepcopy(global_model)
    tgt_model_list = get_model_list(global_model=tgt_global_model, dataloader=tgt_dataloader, local_epochs=attack_epochs, logger=logger, rnd=rnd)
    atk_model_list = get_model_list(global_model=atk_global_model, dataloader=atk_dataloader, local_epochs=attack_epochs, logger=logger, rnd=rnd)    

    # # 获取tgt和atk的损失轨迹，其中atk损失轨迹能区分成员与非成员
    # loss_trace 分为成员与非成员，前1k个为成员，后1k个为非成员
    tgt_loss_trace_mem, tgt_loss_trace_nmem = L_Trace_A(tgt_dataloader_list, tgt_model_list, distillation_model_name, distillation_epochs, logger=logger, rnd = rnd)
    atk_loss_trace_mem, atk_loss_trace_nmem = L_Trace_A(atk_dataloader_list, atk_model_list, distillation_model_name, distillation_epochs, logger=logger, rnd = rnd)
        
    # 处理损失轨迹
    tgt_loss_trace_mem = deal_loss_trace(tgt_loss_trace_mem)
    tgt_loss_trace_nmem = deal_loss_trace(tgt_loss_trace_nmem)
    atk_loss_trace_mem = deal_loss_trace(atk_loss_trace_mem)
    atk_loss_trace_nmem = deal_loss_trace(atk_loss_trace_nmem)
        
    print(f'tgt_loss_trace.shape is {tgt_loss_trace_mem.shape}')
    print(f'atk_loss_trace.shape is {atk_loss_trace_mem.shape}')
    np.save(f'./npy/non-distillation/atk_loss_trace_mem_{rnd}_{attack_epochs}.npy', atk_loss_trace_mem.numpy())
    np.save(f'./npy/non-distillation/atk_loss_trace_nmem_{rnd}_{attack_epochs}.npy', atk_loss_trace_nmem.numpy())
    np.save(f'./npy/non-distillation/tgt_loss_trace_mem_{rnd}_{attack_epochs}.npy', tgt_loss_trace_mem.numpy())
    np.save(f'./npy/non-distillation/tgt_loss_trace_nmem_{rnd}_{attack_epochs}.npy', tgt_loss_trace_nmem.numpy())




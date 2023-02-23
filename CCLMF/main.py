import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import json
import time
import argparse
from utils.utils import *
from models.NCDM.NCD_CD import *
from models.NCDM.NCD_meta import *

# 命令控制语句
argparser = argparse.ArgumentParser(description="diff through pp")
argparser.add_argument('--epochs', default=20, type=int, 
                    metavar='N',
                    help='number of total epochs to run')
argparser.add_argument('-lr', '--learning-rate', default=1e-3, type=float,
                metavar='LR', help='initial learning rate', dest='lr')
argparser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N')
argparser.add_argument('-g','--gpu', type=str, default='cuda')
argparser.add_argument('-i', '--index', default=0, type=int)
argparser.add_argument('-r', '--ratio', default=0, type=float)
argparser.add_argument('-n', '--number', default=30, type=int)
argparser.add_argument('-dn', '--dataset-name', default="java-30", type=str)
argparser.add_argument('-mn', '--model-name', default="NCD_meta", type=str)
argparser.add_argument('-hz', '--hidden-size', default=512, type=int)
args = argparser.parse_args()

# 一些参数
model_name = args.model_name
batch_size = args.batch_size
epoch = args.epochs
hidden_size = args.hidden_size
ratio = args.ratio
index = args.index
device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
dataset_name = args.dataset_name
pre_dataset_name = "baiteng_Python程序设计-924247594312409088"
pre_model = "./python_model"

# 记录文件
logger = Logger()
def log(str):
    print(str)
    logger.log(str + '\n')
current_time = time.strftime('%Y-%m-%d-%H:%M', time.localtime())
log_dir = f'./logs/emcdr/{model_name}/{args.number}/'
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
logger.set_logdir(log_dir)
log_name = f"{model_name}_{current_time}_{ratio}.log"
logger.set_filename(log_name)
log_name = log_dir+log_name

log(f"start training {model_name} on {dataset_name}, batchsize: {batch_size}, epoch: {epoch}, lr: {args.lr}, device: {device}, hidden_size: {hidden_size}")
log(f"log file saved in {log_name}")
log(f"pre_model_name: {pre_model}, drop_ratio: {ratio}")


def transform(user, item, item2knowledge, score, batch_size):
    knowledge_emb = torch.zeros((len(item), knowledge_n))
    for idx in range(len(item)):
        knowledge_emb[idx][np.array(item2knowledge[item[idx]])] = 1.0

    data_set = TensorDataset(
        torch.tensor(user, dtype=torch.int64),
        torch.tensor(item, dtype=torch.int64),
        knowledge_emb,
        torch.tensor(score, dtype=torch.float32)
    )
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)

# 获取目标域的Q矩阵
df_item = pd.read_csv(f"../baiteng/dataset/final/{dataset_name}/item.csv")
item2knowledge = {}
knowledge_set = set()
for i, s in df_item.iterrows():
    item_id, knowledge_codes = s['item_id'], s['knowledge_code']
    item2knowledge[item_id] = knowledge_codes

# 读取源域和目标域的参数信息
src_info = f"../baiteng/dataset/final/{pre_dataset_name}/info.json"
dst_info = f"../baiteng/dataset/final/{dataset_name}/info.json"
with open(src_info, 'r') as f:
    info = json.load(f)
pre_user_n = info['student_cnt']
pre_item_n = info['problem_cnt']
pre_knowledge_n = info['concept_cnt']
log(f"src_data: user_n: {pre_user_n}, item_n: {pre_item_n}, konwledge_n: {pre_knowledge_n}")
with open(dst_info, 'r') as f:
    info = json.load(f)
user_n = info['student_cnt']
item_n = info['problem_cnt']
knowledge_n = info['concept_cnt']
log(f"dst_data: user_n: {user_n}, item_n: {item_n}, konwledge_n: {knowledge_n}")

# 读取测试集
test_data = pd.read_csv(f"../baiteng/dataset/final/{dataset_name}/test.csv")
test_set = transform(test_data["user_id"], test_data["item_id"], item2knowledge, test_data["score"], batch_size)
log(f"transform test_data done.")

for ratio in [ 0.3, 0.4, 0.5]:
    log(f"ratio: {ratio}")
    a = range(10)
    if ratio==0:
        a = [0]

    for index in a:
        log(f"begin trainning with ratio:{ratio}, index: {index}")
        # 读取数据集
        dir_path = f"../baiteng/dataset/final/{dataset_name}/"
        # test_data = pd.read_csv(dir_path+"test1.csv")
        # train_data = test_data
        # valid_data = test_data
        if ratio!=0:
            dir_path = f"../baiteng/dataset/final/{dataset_name}/{ratio}/{index}/"
        train_data = pd.read_csv(dir_path+"train.csv")
        valid_data = pd.read_csv(dir_path+"valid.csv")
        log(f"data_dir: {dir_path}")

        train_set = transform(train_data["user_id"], train_data["item_id"], item2knowledge, train_data["score"], batch_size)
        log(f"transform train_data done.")
        valid_set = transform(valid_data["user_id"], valid_data["item_id"], item2knowledge, valid_data["score"], batch_size)
        log(f"transform valid_data done.")

        best_model = f"./logs/emcdr/{model_name}/{args.number}/best/{ratio}_{index}_best_model"
        if model_name=="NCD_CD":
            cdm = NCD_CD(pre_knowledge_n, pre_item_n, pre_user_n, knowledge_n, item_n, user_n, pre_model, log_name, device, best_model)
        if model_name=="NCD_meta":
            cdm = NCD_meta(pre_knowledge_n, pre_item_n, pre_user_n, knowledge_n, item_n, user_n, pre_model, log_name, device, best_model, hidden_size=hidden_size)
        cdm.train(train_set, valid_set, epoch=epoch, device=device, lr = args.lr)

        cdm.load(best_model)
        auc, accuracy, rmse = cdm.eval(test_set, device=device)
        log("on test data: auc: %.6f, accuracy: %.6f, rmse: %.6f" % (auc, accuracy, rmse))


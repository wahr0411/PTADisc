import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import json
import time
import os
import argparse
from model import NCD, CC_NCD

from utils import Logger
logger = Logger()
def log(str):
    print(str)
    logger.log(str + '\n')


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

if __name__ == '__main__':
    # argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-e', '--epoch', default=30, type=int)
    argparser.add_argument('-lr', '--learning-rate', default=1e-3, type=float)
    argparser.add_argument('-b', '--batch-size', default=64, type=int)
    argparser.add_argument('-g','--gpu', type=str, default='cuda')
    argparser.add_argument('-i', '--index', default=0, type=int)
    argparser.add_argument('-r', '--ratio', default=0, type=float)
    argparser.add_argument('-mn', '--model-name', default="CC_NCD", type=str)
    args = argparser.parse_args()

    # parameters
    model_name = args.model_name
    batch_size = args.batch_size
    epoch = args.epoch
    ratio = args.ratio
    index = args.index
    learning_rate = args.learning_rate
    device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
    pre_model = "./model/ncd_python_model"

    # set log files
    current_time = time.strftime('%Y-%m-%d-%H:%M', time.localtime())
    log_dir = f'./logs/{model_name}/'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_name = log_dir+f"{model_name}_{current_time}_{ratio}.log"
    logger.set_filename(log_name)

    log(f"start training {model_name}, batchsize: {batch_size}, epoch: {epoch}, lr: {learning_rate}, device: {device}")
    log(f"log file saved in {log_name}")
    log(f"pre_model_name: {pre_model}, drop_ratio: {ratio}")

    # get q-matrix in target course
    df_item = pd.read_csv(f"./data/java-30/item.csv")
    item2knowledge = {}
    knowledge_set = set()
    for i, s in df_item.iterrows():
        item_id, knowledge_codes = s['item_id'], list(set(eval(s['knowledge_code'])))
        item2knowledge[item_id] = knowledge_codes
        knowledge_set.update(knowledge_codes)

    # get info in source and target course
    pre_user_n = 29454
    pre_item_n = 17787
    pre_knowledge_n = 685
    log(f"src_data: user_n: {pre_user_n}, item_n: {pre_item_n}, konwledge_n: {pre_knowledge_n}")
    dst_info = f"./data/java-30/info.json"
    with open(dst_info, 'r') as f:
        info = json.load(f)
    user_n = info['student_cnt']
    item_n = info['problem_cnt']
    knowledge_n = info['concept_cnt']
    log(f"dst_data: user_n: {user_n}, item_n: {item_n}, konwledge_n: {knowledge_n}")

    # read test dataset
    test_data = pd.read_csv(f"./data/java-30/test.csv")
    test_set = transform(test_data["user_id"], test_data["item_id"], item2knowledge, test_data["score"], batch_size)
    log(f"transform test_data done.")

    # index array
    a = range(10)
    if ratio==0:
        a = [0]

    for index in a:
        log(f"begin trainning with ratio:{ratio}, index: {index}")
        # read train and valid dataset
        dir_path = f"./data/java-30/"
        if ratio!=0:
            dir_path = f"./data/java-30/{ratio}/{index}/"
        train_data = pd.read_csv(dir_path+"train.csv")
        valid_data = pd.read_csv(dir_path+"valid.csv")
        log(f"data_dir: {dir_path}")

        train_set = transform(train_data["user_id"], train_data["item_id"], item2knowledge, train_data["score"], batch_size)
        log(f"transform train_data done.")
        valid_set = transform(valid_data["user_id"], valid_data["item_id"], item2knowledge, valid_data["score"], batch_size)
        log(f"transform valid_data done.")

        model_dir =  f"./model/{model_name}/"
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        best_model = model_dir+f"{current_time}_{ratio}_{index}_best_model"
        if model_name=="NCD":
            cdm = NCD(knowledge_n, item_n, user_n, log_name, best_model)
        if model_name=="CC_NCD":
            cdm = CC_NCD(pre_knowledge_n, pre_item_n, pre_user_n, knowledge_n, item_n, user_n, pre_model, log_name, device, best_model)
        cdm.train(train_set, valid_set, epoch=epoch, device=device, lr = learning_rate)

        cdm.load(best_model)
        auc, accuracy, rmse = cdm.eval(test_set, device=device)
        log("on test data: auc: %.6f, accuracy: %.6f, rmse: %.6f" % (auc, accuracy, rmse))


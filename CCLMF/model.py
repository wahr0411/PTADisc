import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from EduCDM import CDM
import time
import os


class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)

class MetaNet(torch.nn.Module):
    def __init__(self, emb_dim1, emb_dim2):
        super().__init__()
        self.decoder = torch.nn.Sequential(torch.nn.Linear(emb_dim1, 100), torch.nn.ReLU(),
                                           torch.nn.Linear(100, emb_dim1 * emb_dim2))

    def forward(self, input):
        output = self.decoder(input)
        return output

class NCDNet(nn.Module):
    def __init__(self, knowledge_n, exer_n, student_n):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n 
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable

        super(NCDNet, self).__init__()

        # prediction sub-net
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_difficulty = nn.Embedding(self.exer_n, 1)
        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, stu_id, input_exercise, input_knowledge_point):
        # before prednet
        stu_emb = self.student_emb(stu_id)
        stat_emb = torch.sigmoid(stu_emb)
        k_difficulty = torch.sigmoid(self.k_difficulty(input_exercise))
        e_difficulty = torch.sigmoid(self.e_difficulty(input_exercise)) 
        # prednet
        input_x = e_difficulty * (stat_emb - k_difficulty) * input_knowledge_point
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))

        return output_1.view(-1)

    def get_embeddings(self, stu_id):
        stu_emb = self.student_emb(stu_id)

        return stu_emb


class NCDNet2(nn.Module):
    def __init__(self, knowledge_n, exer_n, student_n):
        super(NCDNet2, self).__init__()

        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n 
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable

        # prediction sub-net
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_difficulty = nn.Embedding(self.exer_n, 1)
        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_emb, input_exercise, input_knowledge_point):
        # before prednet
        stat_emb = torch.sigmoid(stu_emb)
        k_difficulty = torch.sigmoid(self.k_difficulty(input_exercise))
        e_difficulty = torch.sigmoid(self.e_difficulty(input_exercise)) 
        # prednet
        input_x = e_difficulty * (stat_emb - k_difficulty) * input_knowledge_point
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))

        return output_1.view(-1)


class CC_NCDNet(nn.Module):
    def __init__(self, pre_knowledge_n, pre_exer_n, pre_student_n, knowledge_n, exer_n, student_n):
        super(CC_NCDNet, self).__init__()
        self.pre_knowledge_n = pre_knowledge_n
        self.knowledge_n = knowledge_n
        # pretrained net
        self.pre_ncdm_net = NCDNet(pre_knowledge_n, pre_exer_n, pre_student_n)
        # mapping net
        self.mapping = MetaNet(pre_knowledge_n, knowledge_n)
        self.ncdm_net = NCDNet2(knowledge_n, exer_n, student_n)

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, input_exercise, input_knowledge_point):
        # get stu_emb from src domain
        stu_emb = self.pre_ncdm_net.get_embeddings(stu_id).unsqueeze(1)
        # mapping [pre_k_num]->[k_num]
        mapping = self.mapping(stu_emb).view(-1, self.pre_knowledge_n, self.knowledge_n)
        stu_emb = torch.bmm(stu_emb, mapping).squeeze(1)
        # target domain prediction
        output = self.ncdm_net(stu_emb, input_exercise, input_knowledge_point)

        return output


class NCD(CDM):
    '''Neural Cognitive Diagnosis Model'''
    def __init__(self, knowledge_n, exer_n, student_n, log_name, best_model):
        super(NCD, self).__init__()
        self.ncdm_net = NCDNet(knowledge_n, exer_n, student_n)
        self.log_name = log_name
        self.best_model = best_model

    def train(self, train_data, test_data=None, epoch=10, device="cpu", lr=0.002, silence=False):
        last_time = time.time()
        best_auc = 0
        best_rmse = 100
        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.train()
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.ncdm_net.parameters(), lr=lr)
        for epoch_i in range(epoch):
            epoch_losses = []
            batch_count = 0
            for batch_data in tqdm(train_data, "Epoch %s" % epoch_i):
                batch_count += 1
                user_id, item_id, knowledge_emb, y = batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                knowledge_emb: torch.Tensor = knowledge_emb.to(device)
                y: torch.Tensor = (y>0.5).float().to(device)
                pred: torch.Tensor = self.ncdm_net(user_id, item_id, knowledge_emb)
                loss = loss_function(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.mean().item())

            now = time.time()
            print(f'[Epoch {epoch_i}] average loss on train: {np.mean(epoch_losses):1.6f} time cost: {(now - last_time):1.6f}')
            with open(self.log_name, 'a') as f:
                f.write(f"[Epoch {epoch_i}] average loss on train: {np.mean(epoch_losses):1.6f} time cost: {now - last_time:1.6f}"+"\n")
            last_time = now


            if test_data is not None:
                auc, accuracy, rmse = self.eval(test_data, device=device)
                if auc > best_auc:
                    best_auc = auc
                    print("Update best auc!")
                    with open(self.log_name, 'a') as f:
                        f.write("Update best auc!")
                # if rmse < best_rmse:
                #     best_rmse = rmse
                #     print("Update best rmse!")
                #     with open(self.log_name, 'a') as f:
                #         f.write("Update best rmse!")
                    self.save(self.best_model)
                print("[Epoch %d] auc: %.6f, accuracy: %.6f, rmse: %.6f" % (epoch_i, auc, accuracy, rmse))
                with open(self.log_name, 'a') as f:
                    f.write("[Epoch %d] auc: %.6f, accuracy: %.6f, rmse: %.6f" % (epoch_i, auc, accuracy, rmse)+"\n")


    def eval(self, test_data, device="cpu", threshold=0.5):
        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.eval()
        y_true, y_pred = [], []
        for batch_data in tqdm(test_data, "Evaluating"):
            user_id, item_id, knowledge_emb, y = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            knowledge_emb: torch.Tensor = knowledge_emb.to(device)
            pred: torch.Tensor = self.ncdm_net(user_id, item_id, knowledge_emb)
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())

        auc = roc_auc_score(np.array(y_true) >= 0.5, y_pred)
        acc = accuracy_score(np.array(y_true) >= 0.5, np.array(y_pred) >= threshold)
        rmse = mean_squared_error(np.array(y_true) >= 0.5, y_pred, squared=False)
        return auc, acc, rmse
        # return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5)

    def save(self, filepath):
        torch.save(self.ncdm_net.state_dict(), filepath)

    def load(self, filepath):
        self.ncdm_net.load_state_dict(torch.load(filepath))  # , map_location=lambda s, loc: s


class CC_NCD(CDM):
    def __init__(self, pre_knowledge_n, pre_exer_n, pre_student_n, knowledge_n, exer_n, student_n, pre_model_name, log_name, device, best_model):
        super(CC_NCD, self).__init__()
        self.ncd_meta_net = CC_NCDNet(pre_knowledge_n, pre_exer_n, pre_student_n, knowledge_n, exer_n, student_n)
        self.load_pre(pre_model_name, device)
        self.log_name = log_name
        self.best_model = best_model

    def train(self, train_data, test_data=None, epoch=10, device="cpu", lr=0.002, silence=False):
        last_time = time.time()
        best_auc = 0
        self.ncd_meta_net = self.ncd_meta_net.to(device)
        self.ncd_meta_net.train()
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.ncd_meta_net.parameters()), lr=lr)
        for epoch_i in range(epoch):
            epoch_losses = []
            batch_count = 0
            for batch_data in tqdm(train_data, "Epoch %s" % epoch_i):
                batch_count += 1
                user_id, item_id, knowledge_emb, y = batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                knowledge_emb: torch.Tensor = knowledge_emb.to(device)
                y: torch.Tensor = (y>0.5).float().to(device)
                pred: torch.Tensor = self.ncd_meta_net(user_id, item_id, knowledge_emb)
                loss = loss_function(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.mean().item())

            now = time.time()
            print(f'[Epoch {epoch_i}] average loss on train: {np.mean(epoch_losses):1.6f} time cost: {(now - last_time):1.6f}')
            with open(self.log_name, 'a') as f:
                f.write(f"[Epoch {epoch_i}] average loss on train: {np.mean(epoch_losses):1.6f} time cost: {now - last_time:1.6f}"+"\n")
            last_time = now


            if test_data is not None:
                auc, accuracy, rmse = self.eval(test_data, device=device)
                if auc > best_auc:
                    best_auc = auc
                    print("Update best auc!")
                    with open(self.log_name, 'a') as f:
                        f.write("Update best auc!")
                    self.save(self.best_model)
                print("[Epoch %d] auc: %.6f, accuracy: %.6f, rmse: %.6f" % (epoch_i, auc, accuracy, rmse))
                with open(self.log_name, 'a') as f:
                    f.write("[Epoch %d] auc: %.6f, accuracy: %.6f, rmse: %.6f" % (epoch_i, auc, accuracy, rmse)+"\n")


    def eval(self, test_data, device="cpu",  threshold=0.5):
        self.ncd_meta_net = self.ncd_meta_net.to(device)
        self.ncd_meta_net.eval()
        y_true, y_pred = [], []
        for batch_data in tqdm(test_data, "Evaluating"):
            user_id, item_id, knowledge_emb, y = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            knowledge_emb: torch.Tensor = knowledge_emb.to(device)
            pred: torch.Tensor = self.ncd_meta_net(user_id, item_id, knowledge_emb)
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())

        auc = roc_auc_score(np.array(y_true) >= 0.5, y_pred)
        acc = accuracy_score(np.array(y_true) >= 0.5, np.array(y_pred) >= threshold)
        rmse = mean_squared_error(np.array(y_true) >= 0.5, y_pred, squared=False)
        return auc, acc, rmse

    def load(self, filepath):
        self.ncd_meta_net.load_state_dict(torch.load(filepath))  # , map_location=lambda s, loc: s

    def load_pre(self, filepath, device):
        self.ncd_meta_net.pre_ncdm_net.load_state_dict(torch.load(filepath, map_location=device))  # , map_location=lambda s, loc: s

    def save(self, filepath):
        torch.save(self.ncd_meta_net.state_dict(), filepath)


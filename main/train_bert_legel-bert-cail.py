# -*- coding: utf-8 -*-
# @Time    : 2024/11/24 15:33
# @Author  : Wenlin Zhong
# @Contact : 13527860108@163.com

from itertools import chain
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import Sequential
import torch.optim as optim
import datetime
import math
from typing import List
from transformers import BertConfig, BertModel

import argparse
import copy
import datetime
import os
import pickle
import sys
import time
import random
import torch
from sklearn.metrics import (f1_score,precision_score, recall_score,accuracy_score)
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from utils.optim import ScheduledOptim
from transformers import AutoTokenizer, BertConfig

import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import hamming_loss, jaccard_score

class LawModel(nn.Module):
    def __init__(self, config):
        super(LawModel, self).__init__()
        self.config = config

        self.bert_config = BertConfig.from_pretrained(config.bert_path, output_hidden_states=False)
        self.bert = BertModel.from_pretrained(config.bert_path, config=self.bert_config)

        for param in self.bert.parameters():
            param.requires_grad = True        
        
        self.law_classifier = torch.nn.Sequential(
            torch.nn.Linear(self.bert_config.hidden_size, config.mlp_size),
            torch.nn.ReLU(),
            torch.nn.Linear(config.mlp_size, config.law_label_size)
        )
        self.law_loss = torch.nn.BCEWithLogitsLoss()  # 使用二元交叉熵损失函数
        
               
    def classifier_layer(self, doc_out, law_labels):
        """
        :param doc_out: [batch_size, 4 * hidden_dim]
        :param law_labels: [batch_size, law_label_size]
        """
        
        law_logits = self.law_classifier(doc_out)  # [batch_size, law_label_size]
        law_loss = self.law_loss(law_logits, law_labels.float())  # 将标签转换为float类型
        law_probs = torch.sigmoid(law_logits)  # 使用sigmoid激活函数
        law_predicts = (law_probs > 0.3).int()  # 使用0.5作为阈值进行预测

        # print("law_predicts:",law_predicts)
        # print("law_labels.float():",law_labels.float())

        return law_predicts, law_loss


    def forward(self, input_facts, type_ids_list, attention_mask_list, law_labels):
        """
        Args:
            input_facts: [batch_size, max_sent_num, max_sent_seq_len]
            type_ids_list: [batch_size, max_sent_num, max_sent_seq_len]
            attention_mask_list: [batch_size, max_sent_num, max_sent_seq_len]
            law_labels: [batch_size, law_label_size]  # 多标签分类任务的标签

        Returns:
            law_loss: 损失值
            law_preds: 预测的法条序号列表
        """
        outputs = self.bert.forward(input_ids=input_facts, attention_mask=attention_mask_list, token_type_ids=type_ids_list)
        doc_rep = outputs.pooler_output
        law_preds, law_loss = self.classifier_layer(doc_rep, law_labels)  # [batch_size, law_label_size]

        return law_loss, law_preds
    
    
    def predict(self, input_facts, type_ids_list, attention_mask_list, law_labels):
        # compute query features
        outputs = self.bert.forward(input_ids=input_facts, attention_mask=attention_mask_list, token_type_ids=type_ids_list)
        doc_rep = outputs.pooler_output
        law_preds, _ = self.classifier_layer(doc_rep, law_labels)  # [batch_size, law_label_size]
        
        return law_preds


os.chdir('/home/u22451152/Uni-LAP/main')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_rand(SEED_NUM):
    torch.random.manual_seed(SEED_NUM)
    torch.manual_seed(SEED_NUM)
    random.seed(SEED_NUM)
    np.random.seed(SEED_NUM)
    torch.cuda.manual_seed(SEED_NUM)
    torch.cuda.manual_seed_all(SEED_NUM)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False

class Config:
    def __init__(self):
        self.MAX_SENTENCE_LENGTH = 500
        self.word_emb_dim = 200
        self.pretrain_word_embedding = None
        self.word2id_dict = None
        self.id2word_dict = None
        self.bert_path = None

        self.law_label_size = 74  # TODO law_label_size
        self.law_relation_threshold = 0.3

        self.sent_len = 100
        self.doc_len = 15
        #  hyperparameters
        self.HP_iteration = 100
        self.HP_batch_size = 64
        self.HP_hidden_dim = 200
        self.HP_dropout = 0.2

        self.HP_gpu = True
        self.HP_lr = 0.015
        self.HP_lr_decay = 0.05
        self.HP_clip = 5.0
        self.HP_momentum = 0
        self.bert_hidden_size = 768
        self.HP_freeze_word_emb = True

        # optimizer
        self.use_adam = False
        self.use_bert = True
        self.use_sgd = False
        self.use_adadelta = False
        self.use_warmup_adam = False
        self.mode = 'train'

        self.save_model_dir = ""
        self.save_dset_dir = ""


        self.confused_matrix = None
        self.mlp_size = 512

        self.seed = 10


    def show_data_summary(self):
        print("DATA SUMMARY START:")
        print("     MAX SENTENCE LENGTH: %s" % (self.MAX_SENTENCE_LENGTH))
        print("     Word embedding size: %s" % (self.word_emb_dim))
        print("     Bert Path:           %s" % (self.bert_path))
        print("     Law label     size:  %s" % (self.law_label_size))

        print("     Hyperpara  iteration: %s" % (self.HP_iteration))
        print("     Hyperpara  batch size: %s" % (self.HP_batch_size))
        print("     Hyperpara          lr: %s" % (self.HP_lr))
        print("     Hyperpara    lr_decay: %s" % (self.HP_lr_decay))
        print("     Hyperpara     HP_clip: %s" % (self.HP_clip))
        print("     Hyperpara    momentum: %s" % (self.HP_momentum))
        print("     Hyperpara  hidden_dim: %s" % (self.HP_hidden_dim))
        print("     Hyperpara     dropout: %s" % (self.HP_dropout))

        print("     Hyperpara         GPU: %s" % (self.HP_gpu))
        print("     Seed                :  %s" % (self.seed))
        print("DATA SUMMARY END.")
        sys.stdout.flush()


    def load_word_pretrain_emb(self, emb_path):
        self.pretrain_word_embedding = np.cast[np.float32](np.load(emb_path))
        self.word_emb_dim = self.pretrain_word_embedding.shape[1]
        print("word embedding size:", self.pretrain_word_embedding.shape)



class BERTDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, id2word_dict, fact_type, law_label_size=74): # TODO law_label_size
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = data
        self.id2word_dict = id2word_dict
        self.fact_type = fact_type
        self.law_label_size = law_label_size  # 法条总数

    def __len__(self):
        if self.fact_type == 'fact':
            return len(self.data['raw_facts_list'])

    def _convert_ids_to_sent(self, fact):
        # 解析 token ID 为文本
        # fact: [max_sent_num, max_sent_len]
        mask = np.array(fact) == 164672
        mask = ~mask
        seq_len = mask.sum(1)  # [max_sent_num]
        sent_num_mask = seq_len == 0
        sent_num_mask = ~sent_num_mask
        sent_num = sent_num_mask.sum(0)
        raw_text = []
        for s_idx in range(sent_num):
            cur_seq_len = seq_len[s_idx]
            raw_text.extend(fact[s_idx][:cur_seq_len])

        return [self.id2word_dict[ids] for ids in raw_text]

    def __getitem__(self, index):
        raw_fact_list = self.data['raw_facts_list'][index]
        law_label_lists = self.data['law_label_lists'][index]

        # 将 law_label_lists 转换为二进制向量
        law_label_vector = torch.zeros(self.law_label_size, dtype=torch.float)
        for law_id in law_label_lists:
            law_label_vector[law_id] = 1.0  # 标记为 1 表示包含该法条

        return raw_fact_list, law_label_vector

    def collate_bert_fn(self, batch):
        batch_raw_fact_list, batch_law_label_lists = [], []

        for item in batch:
            batch_raw_fact_list.append(item[0])
            batch_law_label_lists.append(item[1])

        # 分词并处理为统一长度（500）
        batch_out = self.tokenizer.batch_encode_plus(
            batch_raw_fact_list, max_length=500, padding='max_length', return_tensors='pt', truncation=True
        )

        padded_input_ids = torch.LongTensor(batch_out['input_ids']).to(DEVICE)
        padded_token_type_ids = torch.LongTensor(batch_out['token_type_ids']).to(DEVICE)
        padded_attention_mask = torch.LongTensor(batch_out['attention_mask']).to(DEVICE)
        padded_law_label_lists = torch.stack(batch_law_label_lists).to(DEVICE)  # 将标签堆叠为张量

        return padded_input_ids, padded_token_type_ids, padded_attention_mask, batch_raw_fact_list, padded_law_label_lists


def load_dataset(path):
    # CAIL数据集
    train_path = os.path.join(path, "train_filtered_cail.pkl")
    valid_path = os.path.join(path, "valid_filtered_cail.pkl")
    test_path = os.path.join(path, "test_filtered_cail.pkl")

    train_dataset = pickle.load(open(train_path, mode='rb'))
    valid_dataset = pickle.load(open(valid_path, mode='rb'))
    test_dataset = pickle.load(open(test_path, mode='rb'))

    print("train dataset sample:", train_dataset['raw_facts_list'][0])
    print("train dataset sample len:", len(train_dataset['law_label_lists']))
    return train_dataset, valid_dataset, test_dataset

def str2bool(params):
    return True if params.lower() == 'true' else False

def save_data_setting(data: Config, save_file):
    new_data = copy.deepcopy(data)
    ## remove input instances
    ## save data settings
    with open(save_file, 'wb') as fp:
        pickle.dump(new_data, fp)
    print("Data setting saved to file: ", save_file)

def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr * ((1 - decay_rate) ** epoch)
    print(" Learning rate is setted as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def load_data_setting(save_file):
    with open(save_file, 'rb') as fp:
        data = pickle.load(fp)
    print("Data setting loaded from file: ", save_file)
    data.show_data_summary()
    return data


def get_result(law_target, law_preds, mode):
    """
    计算多标签分类任务的评估指标。
    :param law_target: 真实标签，二维列表或数组（样本数 × 法条数）
    :param law_preds: 预测标签，二维列表或数组（样本数 × 法条数）
    :param mode: 评估模式（用于打印日志）
    :return: 宏平均 F1 分数
    """
    # 计算宏平均 F1、Precision 和 Recall
    law_macro_f1 = f1_score(law_target, law_preds, average="macro", zero_division=0)
    law_macro_precision = precision_score(law_target, law_preds, average="macro", zero_division=0)
    law_macro_recall = recall_score(law_target, law_preds, average="macro", zero_division=0)
    law_accuracy_score = accuracy_score(law_target, law_preds)
    
    # 计算 Hamming Loss 和 Jaccard Similarity
    hamming = hamming_loss(law_target, law_preds)
    jaccard = jaccard_score(law_target, law_preds, average="samples", zero_division=0)

    print(f"Law task ({mode}):")
    print(f"@ACC: {law_accuracy_score:.4f} Macro F1: {law_macro_f1:.4f}, Macro Precision: {law_macro_precision:.4f}, Macro Recall: {law_macro_recall:.4f}")
    print(f"@Hamming Loss: {hamming:.4f}, Jaccard Similarity: {jaccard:.4f}")

    return law_macro_f1


def evaluate(model, valid_dataloader, name, epoch_idx):
    """
    评估模型在多标签分类任务上的性能。
    :param model: 模型
    :param valid_dataloader: 验证集 DataLoader
    :param name: 评估名称（用于打印日志）
    :param epoch_idx: 当前 epoch 索引
    :return: 宏平均 F1 分数
    """
    model.eval()
    ground_law_y = []  # 真实标签
    predicts_law_y = []  # 预测标签

    print_num = 0

    for batch_idx, datapoint in enumerate(valid_dataloader):
        fact_list, type_ids_list, attention_mask_list, _, law_label_lists = datapoint

        # 获取模型预测结果
        law_preds = model.predict(fact_list, type_ids_list, attention_mask_list, law_label_lists)

        # if print_num <= 2:
        #     print("law_preds",law_preds[:5])
        #     print("law_label_lists",law_label_lists[:5])
        # 收集真实标签和预测标签
        ground_law_y.extend(law_label_lists.cpu().tolist())
        predicts_law_y.extend(law_preds.cpu().tolist())
        print_num += 1
    # 计算评估指标
    score = get_result(ground_law_y, predicts_law_y, name)

    return score


def train(model, dataset, config: Config):
    train_dataloader = dataset["train_data_set"]
    valid_dataloader = dataset["valid_data_set"]
    test_dataloader = dataset["test_data_set"]
    print("config batch size:", config.HP_batch_size)
    print("Training model...")
    print(model)
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    if config.use_warmup_adam:
        optimizer = ScheduledOptim(optim.Adam(parameters, betas=(0.9, 0.98), eps=1e-9), d_model=256, n_warmup_steps=2000)
    elif config.use_sgd:
        optimizer = optim.SGD(parameters, lr=config.HP_lr, momentum=config.HP_momentum)
    elif config.use_adam:
        optimizer = optim.Adam(parameters, lr=config.HP_lr)
    elif config.use_bert:
        print("optimizer use_bert!")
        optimizer = optim.Adam(parameters, lr=5e-6)  # fine tuning
    else:
        raise ValueError("Unknown optimizer")
    print('optimizer: ', optimizer)

    for idx in range(config.HP_iteration):
        epoch_start = time.time()
        temp_start = epoch_start
        print("Epoch: %s/%s" % (idx, config.HP_iteration))
        sample_law_loss =  0

        model.train()
        model.zero_grad()

        batch_size = config.HP_batch_size

        ground_law_y, predicts_law_y = [], []

        for batch_idx, datapoint in enumerate(train_dataloader):
            fact_list, type_ids_list, attention_mask_list, _, law_label_lists = datapoint
            law_loss, law_preds = model.forward(fact_list, type_ids_list, attention_mask_list, law_label_lists)

            loss = law_loss

            sample_law_loss += law_loss.data

            # 将 logits 转换为二进制向量
            ground_law_y.extend(law_label_lists.cpu().tolist())
            predicts_law_y.extend(law_preds.cpu().tolist())

            if (batch_idx + 1) % 100 == 0:
                # 计算 Hamming Loss 和 Jaccard Similarity
                cur_hamming = hamming_loss(ground_law_y, predicts_law_y)
                cur_jaccard = jaccard_score(ground_law_y, predicts_law_y, average="samples", zero_division=0)

                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time

                print("Instance: %s; Time: %.2fs; law loss: %.2f; Hamming Loss: %.4f; Jaccard: %.4f" %
                      ((batch_idx + 1), temp_cost, sample_law_loss, cur_hamming, cur_jaccard))
                
                if (batch_idx + 1) % 1000 == 0:
                    current_score = evaluate(model, valid_dataloader, "Valid", -1)

                sys.stdout.flush()
                sample_law_loss = 0

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.HP_clip)
            optimizer.step()
            model.zero_grad()

        sys.stdout.flush()

        # 验证集评估
        current_score = evaluate(model, valid_dataloader, "Valid", -1)
        print(f"dev current score: {current_score}")

        # 保存模型
        model_name = os.path.join(config.save_model_dir, f"{idx}.ckpt")
        torch.save(model.state_dict(), model_name)

        # 测试集评估
        _ = evaluate(model, test_dataloader, "Test", -1)


def Test(model, dataset, config: Config):
    test_dataloader = dataset["test_data_set"]
    print(model)
    _ = evaluate(model, test_dataloader, "Test", -1)

if __name__ == '__main__':
    print(datetime.datetime.now())
    BASE = "/home/u22451152/Uni-LAP/main"
    parser = argparse.ArgumentParser(description='Uni-LAP')
    parser.add_argument('--data_path', default="/home/u22451152/Uni-LAP/main/datasets/cail")
    parser.add_argument('--status', default="test")
    parser.add_argument('--savemodel', default=BASE+"/results/cail/legal-bert")
    parser.add_argument('--loadmodel', default="")

    parser.add_argument('--embedding_path', default=BASE+'/cail_thulac.npy')
    parser.add_argument('--word2id_dict', default=BASE+'/data/w2id_thulac.pkl')

    parser.add_argument('--word_emb_dim', default=200, type=int)
    parser.add_argument('--MAX_SENTENCE_LENGTH', default=510, type=int)

    parser.add_argument('--HP_iteration', default=16, type=int)
    parser.add_argument('--HP_batch_size', default=32, type=int)
    parser.add_argument('--HP_hidden_dim', default=256, type=int)
    parser.add_argument('--HP_dropout', default=0.2, type=float)

    parser.add_argument('--HP_lr', default=1e-3, type=float)
    parser.add_argument('--HP_lr_decay', default=0.05, type=float)
    parser.add_argument('--HP_freeze_word_emb', action='store_true')

    parser.add_argument('--seed', default=2022, type=int)

    # crime-bert xs刑事
    # parser.add_argument('--bert_path', default='/home/u22451152/Uni-LAP/main/xs', type=str)
    # 经典bert
    # parser.add_argument('--bert_path', default='/home/u22451152/google-bert/bert-base-chinese', type=str)
    # legal-bert
    parser.add_argument('--bert_path', default='/home/u22451152/nlpaueb/legal-bert-base-uncased', type=str)

    parser.add_argument('--sample_size', default='all', type=str)

    parser.add_argument('--mlp_size', default=512, type=int)
    parser.add_argument('--law_relation_threshold', default=0.3, type=float)
    parser.add_argument('--model_path', 
                        default='/home/u22451152/Uni-LAP/main/results/cail/legal-bert/2025-01-07 14:18:23.435001/15.ckpt', 
                        type=str)
    args = parser.parse_args()
    print(args)

    seed_rand(args.seed)

    status = args.status
    
    print('New config....')
    config = Config()
    config.HP_iteration = args.HP_iteration
    config.HP_batch_size = args.HP_batch_size
    config.HP_hidden_dim = args.HP_hidden_dim
    config.HP_dropout = args.HP_dropout
    config.HP_lr = args.HP_lr
    config.MAX_SENTENCE_LENGTH = args.MAX_SENTENCE_LENGTH
    config.HP_lr_decay = args.HP_lr_decay
    config.save_model_dir = os.path.join(args.savemodel, f"{datetime.datetime.now()}")
    config.HP_freeze_word_emb = args.HP_freeze_word_emb
    if not os.path.exists(config.save_model_dir):
        os.makedirs(config.save_model_dir)

    config.mlp_size = args.mlp_size
    config.word2id_dict = pickle.load(open(args.word2id_dict, 'rb'))
    config.id2word_dict = {item[1]: item[0] for item in config.word2id_dict.items()}
    config.bert_path = args.bert_path
    config.seed = args.seed

    config.load_word_pretrain_emb(args.embedding_path)
    save_data_setting(config, os.path.join(config.save_model_dir, 'data.dset'))
    config.show_data_summary()

    print("\nLoading data...")
    tokenizer = AutoTokenizer.from_pretrained(args.bert_path)  # 导入分词器
    train_data, valid_data, test_data = load_dataset(args.data_path)
    if args.sample_size != 'all':
        sample_size = int(args.sample_size)
        sampled_train_data = {}
        start = random.randint(0, len(train_data['raw_facts_list']) - sample_size)
        print("start:", start)
        for k, v in train_data.items():
            sampled_train_data[k] = train_data[k][start:start + sample_size]

        train_data = sampled_train_data
        
    if status == 'train':
        
        train_dataset = BERTDataset(train_data, tokenizer, config.MAX_SENTENCE_LENGTH, config.id2word_dict, 'fact')
        valid_dataset = BERTDataset(valid_data, tokenizer, config.MAX_SENTENCE_LENGTH, config.id2word_dict, 'fact')
        test_dataset = BERTDataset(test_data, tokenizer, config.MAX_SENTENCE_LENGTH, config.id2word_dict, 'fact')

        train_dataloader = DataLoader(train_dataset, batch_size=config.HP_batch_size, shuffle=True, collate_fn=train_dataset.collate_bert_fn)
        valid_dataloader = DataLoader(valid_dataset, batch_size=config.HP_batch_size, shuffle=False, collate_fn=valid_dataset.collate_bert_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=config.HP_batch_size, shuffle=False, collate_fn=test_dataset.collate_bert_fn)

        print("train_data %d, valid_data %d, test_data %d." % (
            len(train_dataset), len(valid_dataset), len(test_dataset)))

        data_dict = {
            "train_data_set": train_dataloader,
            "test_data_set": test_dataloader,
            "valid_data_set": valid_dataloader
        }

        seed_rand(args.seed)
        model = LawModel(config)

        # model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))

        # 训练阶段
        print("\nTraining...")
        if config.HP_gpu:
            model.cuda()
        train(model, data_dict, config)

    elif status == 'test':
        train_dataset = BERTDataset(train_data, tokenizer, config.MAX_SENTENCE_LENGTH, config.id2word_dict, 'fact')
        valid_dataset = BERTDataset(valid_data, tokenizer, config.MAX_SENTENCE_LENGTH, config.id2word_dict, 'fact')
        test_dataset = BERTDataset(test_data, tokenizer, config.MAX_SENTENCE_LENGTH, config.id2word_dict, 'fact')

        train_dataloader = DataLoader(train_dataset, batch_size=config.HP_batch_size, shuffle=True, collate_fn=train_dataset.collate_bert_fn)
        valid_dataloader = DataLoader(valid_dataset, batch_size=config.HP_batch_size, shuffle=False, collate_fn=valid_dataset.collate_bert_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=config.HP_batch_size, shuffle=False, collate_fn=test_dataset.collate_bert_fn)

        print("test_data %d." % (len(test_dataset)))

        data_dict = {"test_data_set": test_dataloader}

        seed_rand(args.seed)
        model = LawModel(config)
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
        if config.HP_gpu:
            model.cuda()

        print("\nTesting...")
        Test(model, data_dict, config)

        _ = evaluate(model, train_dataloader, "train_dataloader", -1)

        _ = evaluate(model, valid_dataloader, "valid_dataloader", -1)
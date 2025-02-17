import csv
import logging
import time
from copy import deepcopy

import math
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from Project.Project3.model.kgrs import KGRS
from pytorch_lightning import seed_everything


def nDCG(sorted_items, pos_item, train_pos_item, k=5):
    dcg = 0
    train_pos_item = set(train_pos_item)
    filter_item = set(filter(lambda item: item not in train_pos_item, pos_item))
    max_correct = min(len(filter_item), k)
    train_hit_num = 0
    valid_num = 0
    recommended_items = set()
    for index in range(len(sorted_items)):
        if sorted_items[index] in train_pos_item:
            train_hit_num += 1
        else:
            valid_num += 1
            if sorted_items[index] in filter_item and sorted_items[index] not in recommended_items:
                dcg += 1 / math.log2(index - train_hit_num + 2)  # Rank starts from 0
                recommended_items.add(sorted_items[index])
            if valid_num >= k:
                break
    idcg = sum([1 / math.log2(i + 2) for i in range(max_correct)])
    return dcg / idcg


def load_data():
    train_pos, train_neg = np.load("../data/train_pos.npy"), np.load("../data/train_neg.npy")
    train_pos_len, train_neg_len = int(len(train_pos) * 0.8), int(len(train_neg) * 0.8)
    test_pos, test_neg = train_pos[train_pos_len:], train_neg[train_neg_len:]
    train_pos, train_neg = train_pos[:train_pos_len], train_neg[:train_neg_len]
    return train_pos, train_neg, test_pos, test_neg


def get_user_pos_items(train_pos, test_pos):
    user_pos_items, user_train_pos_items = {}, {}
    for record in train_pos:
        user, item = record[0], record[1]
        if user not in user_train_pos_items:
            user_train_pos_items[user] = set()
        user_train_pos_items[user].add(item)
    for record in test_pos:
        user, item = record[0], record[1]
        if user not in user_train_pos_items:
            user_train_pos_items[user] = set()
        if user not in user_pos_items:
            user_pos_items[user] = set()
        user_pos_items[user].add(item)
    return user_pos_items, user_train_pos_items


def evaluate():
    train_pos, train_neg, test_pos, test_neg = load_data()
    user_pos_items, user_train_pos_items = get_user_pos_items(train_pos=train_pos, test_pos=test_pos)
    logging.disable(logging.INFO)
    seed_everything(1088, workers=True)
    torch.set_num_threads(8)
    start_time, init_time, train_time, ctr_time, top_k_time = time.time(), 0, 0, 0, 0
    kgrs = KGRS(train_pos=deepcopy(train_pos),
                train_neg=deepcopy(train_neg),
                kg_lines=open('../data/kg.txt', encoding='utf-8').readlines())
    init_time = time.time() - start_time

    kgrs.training()
    train_time = time.time() - start_time - init_time

    test_data = np.concatenate((deepcopy(test_neg), deepcopy(test_pos)), axis=0)
    np.random.shuffle(test_data)
    test_label = test_data[:, 2]
    test_data = test_data[:, :2]
    scores = kgrs.eval_ctr(test_data=test_data)
    auc = roc_auc_score(y_true=test_label, y_score=scores)
    ctr_time = time.time() - start_time - init_time - train_time

    users = list(user_pos_items.keys())
    user_item_lists = kgrs.eval_topk(users=users)
    ndcg5 = np.mean([nDCG(user_item_lists[index], user_pos_items[user], user_train_pos_items[user]) for index, user in
                     enumerate(users)])

    top_k_time = time.time() - start_time - init_time - train_time - ctr_time
    return auc, ndcg5, init_time, train_time, ctr_time, top_k_time, kgrs.config


def main():
    start = time.time()
    auc, ndcg5, init_time, train_time, ctr_time, topk_time, config = evaluate()
    print(config)
    print(f"auc={auc}, ndcg5={ndcg5}")
    print(f"init_time={init_time}, train_time={train_time}, ctr_time={ctr_time}, topk_time={topk_time}")
    print(f"total_time={time.time() - start}")

    filename = '../test/records.csv'
    metrics_data = {'auc': auc, 'ndcg5': ndcg5, 'init_time': init_time, 'train_time': train_time, 'ctr_time': ctr_time,
                    'topk_time': topk_time, 'total_time': time.time() - start}
    with open(filename, mode='a', newline='') as f:
        csvwriter = csv.DictWriter(f, fieldnames=['auc', 'ndcg5', 'total_time', 'batch_size', 'weight_decay',
                                                  'margin',
                                                  'train_time', 'neg_rate', 'ctr_time', 'eval_batch_size', 'emb_dim',
                                                  'l1',
                                                  'learning_rate', 'init_time', 'epoch_num', 'topk_time',
                                                  'hard_neg_ratio'])
        combined_data = {**config, **metrics_data}
        csvwriter.writerow(combined_data)


if __name__ == '__main__':
    main()

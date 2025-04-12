"""
@Author: Du Yunhao
@Filename: AFLink.py
@Contact: dyh_bupt@163.com
@Time: 2021/12/28 19:55
@Discription: Appearance-Free Post Link
"""
import os
import glob
import torch
import numpy as np
from os.path import join, exists
from collections import defaultdict
from sklearn.preprocessing import normalize
from scipy.optimize import linear_sum_assignment
import sys
sys.path.append('E:\\shuju2\\ryjs\\daima\\YOLOv9_DeepSORT-main\\YOLOv9_DeepSORT-main\\yolov9\\AFLink')
import config as cfg
from .train import train
from dataset import LinkData
from .model import PostLinker
INFINITY = 1e5

class AFLink:
    def __init__(self, model, dataset, thrT: tuple, thrS: int, thrP: float):
        # 移除了路径相关的参数，直接使用模型和数据集对象
        self.thrP = thrP
        self.thrT = thrT
        self.thrS = thrS
        self.model = model
        self.dataset = dataset
        self.model.cuda()
        self.model.eval()

    def gather_info(self, track_data):
        id2info = defaultdict(list)
        for row in track_data:
            f, i, x, y, w, h = row
            id2info[i].append([f, x, y, w, h])
        id2info = {k: np.array(v) for k, v in id2info.items()}
        return id2info

    def compression(self, cost_matrix, ids):
        mask_row = cost_matrix.min(axis=1) < self.thrP
        matrix = cost_matrix[mask_row, :]
        ids_row = ids[mask_row]
        mask_col = cost_matrix.min(axis=0) < self.thrP
        matrix = matrix[:, mask_col]
        ids_col = ids[mask_col]
        return matrix, ids_row, ids_col

    def predict(self, track1, track2):
        track1, track2 = self.dataset.transform(track1, track2)
        track1, track2 = track1.unsqueeze(0).cuda(), track2.unsqueeze(0).cuda()
        score = self.model(track1, track2)[0, 1].detach().cpu().numpy()
        return 1 - score

    def predict_batch(self, track_pairs):
        batch_x1, batch_x2 = zip(*[self.dataset.transform(tr1, tr2) for tr1, tr2 in track_pairs])
        batch_x1 = torch.stack(batch_x1).cuda()
        batch_x2 = torch.stack(batch_x2).cuda()
        scores = self.model(batch_x1, batch_x2).detach().cpu().numpy()
        return 1 - scores[:, 1]  # 假设第二列是我们需要的分数
    @staticmethod
    def deduplicate(tracks):
        _, index = np.unique(tracks[:, :2], return_index=True, axis=0)
        return tracks[index]

    # 主函数
    def link(self, track_data):
        id2info = self.gather_info(track_data)
        num = len(id2info)
        ids = np.array(list(id2info))
        cost_matrix = np.ones((num, num)) * INFINITY

        track_pairs = []
        for i, id_i in enumerate(ids):
            for j, id_j in enumerate(ids):
                if id_i == id_j or not self.thrT[0] <= id2info[id_j][0][0] - id2info[id_i][-1][0] < self.thrT[1]:
                    continue
                if self.thrS < np.linalg.norm(id2info[id_i][-1][1:3] - id2info[id_j][0][1:3]):
                    continue
                track_pairs.append((id2info[id_i], id2info[id_j]))
                if len(track_pairs) == 32:
                    costs = self.predict_batch(track_pairs)
                    for pair_index, (track_i, track_j) in enumerate(track_pairs):
                        idx_i_list = np.where(ids == track_i[0])[0]
                        idx_j_list = np.where(ids == track_j[0])[0]
                        if idx_i_list.size > 0 and idx_j_list.size > 0:
                            idx_i = idx_i_list[0]
                            idx_j = idx_j_list[0]
                            cost_matrix[idx_i, idx_j] = costs[pair_index]
                    track_pairs = []

        if track_pairs:
            costs = self.predict_batch(track_pairs)
            for pair_index, (track_i, track_j) in enumerate(track_pairs):
                idx_i_list = np.where(ids == track_i[0])[0]
                idx_j_list = np.where(ids == track_j[0])[0]
                if idx_i_list.size > 0 and idx_j_list.size > 0:
                    idx_i = idx_i_list[0]
                    idx_j = idx_j_list[0]
                    cost_matrix[idx_i, idx_j] = costs[pair_index]

        id2id = {}
        cost_matrix, ids_row, ids_col = self.compression(cost_matrix, ids)
        indices = linear_sum_assignment(cost_matrix)
        for i, j in zip(indices[0], indices[1]):
            if cost_matrix[i, j] < self.thrP:
                id2id[ids_row[i]] = ids_col[j]

        ID2ID = {v: k for k, v in id2id.items()}
        res = track_data.copy()
        for i, track in enumerate(res):
            if track[1] in ID2ID:
                res[i, 1] = ID2ID[track[1]]

        return self.deduplicate(res)





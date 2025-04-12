"""
@Author: Du Yunhao
@Filename: train.py
@Contact: dyh_bupt@163.com
@Time: 2021/12/28 15:04
@Discription: train
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd
import seaborn as sns
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.optim import Adam
from datetime import datetime
from os.path import join, exists
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import precision_score, recall_score, f1_score

import config as cfg
from yolov9.AFLink.model import PostLinker
from yolov9.AFLink.dataset import LinkData

def validate(model, dataloader, loss_fn):
    model.eval()
    loss_sum = 0
    all_labels = []
    all_probabilities = []
    labels = []
    outputs = []

    with torch.no_grad():
        for i, (pair1, pair2, pair3, pair4, label) in enumerate(dataloader):
            pairs_1 = torch.cat((pair1[0], pair2[0], pair3[0], pair4[0]), dim=0).cuda()
            pairs_2 = torch.cat((pair1[1], pair2[1], pair3[1], pair4[1]), dim=0).cuda()
            label = torch.cat(label, dim=0).cuda()
            output = model(pairs_1, pairs_2)
            probabilities = torch.softmax(output, dim=1)[:, 1]  # 获取属于类别1的概率
            loss = loss_fn(output, label)
            loss_sum += loss.item()

            predicted_labels = torch.max(output, 1)[1].cpu().numpy()
            all_labels.extend(label.cpu().numpy())
            labels.extend(label.cpu().numpy())  # 实际标签
            outputs.extend(predicted_labels)  # 预测标签
            all_probabilities.extend(probabilities.cpu().numpy())

    avg_loss = loss_sum / len(dataloader)
    precision = precision_score(labels, outputs, average='macro', zero_division=0)
    recall = recall_score(labels, outputs, average='macro')
    f1 = f1_score(labels, outputs, average='macro', zero_division=0)

    return avg_loss, precision, recall, f1, all_probabilities


def train(save: bool):
    model = PostLinker()
    model.cuda()
    model.train()
    dataset = LinkData(cfg.root_train, 'train')
    dataloader = DataLoader(dataset=dataset, batch_size=cfg.train_batch, shuffle=True, num_workers=cfg.num_workers, drop_last=True)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=cfg.train_lr, weight_decay=cfg.train_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.train_epoch, eta_min=1e-5)

    # Lists for storing metrics
    train_losses = []
    val_losses = []
    val_f1s = []
    val_precisions = []
    val_recalls = []
    all_epoch_probabilities = []  # 存储每个epoch的概率分布数据
    csv_save_path = join(cfg.model_savedir, 'epoch_probabilities.csv')  # Path to save the CSV file
    print('======================= Start Training =======================')
    for epoch in range(cfg.train_epoch):
        print(f'Epoch: {epoch} with lr={optimizer.param_groups[0]["lr"]:.0e}')
        loss_sum = 0
        for i, (pair1, pair2, pair3, pair4, label) in enumerate(dataloader):
            optimizer.zero_grad()
            pairs_1 = torch.cat((pair1[0], pair2[0], pair3[0], pair4[0]), dim=0).cuda()
            pairs_2 = torch.cat((pair1[1], pair2[1], pair3[1], pair4[1]), dim=0).cuda()
            label = torch.cat(label, dim=0).cuda()
            output = model(pairs_1, pairs_2)
            loss = loss_fn(output, label)
            loss_sum += loss.item()
            loss.backward()
            optimizer.step()
        train_losses.append(loss_sum / len(dataloader))

        # Validate and record the metrics
        print('Validating...')
        val_dataset = LinkData(cfg.root_train, 'val')
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=cfg.val_batch, shuffle=False, num_workers=cfg.num_workers, drop_last=False)
        val_loss, val_precision, val_recall, val_f1, all_probabilities = validate(model, val_dataloader, loss_fn)
        val_losses.append(val_loss)
        val_f1s.append(val_f1)
        val_precisions.append(val_precision)
        all_epoch_probabilities.append(all_probabilities)  # 收集每个epoch的概率数据
        val_recalls.append(val_recall)
        scheduler.step()
        print(f'Epoch {epoch} - Training Loss: {train_losses[-1]:.4f}, Validation Loss: {val_loss:.4f}, '
              f'Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}')

        # Save the model after the last epoch
        if save and epoch == cfg.train_epoch - 1:
            if not exists(cfg.model_savedir):
                os.mkdir(cfg.model_savedir)
            torch.save(model.state_dict(), join(cfg.model_savedir, f'newmodel_epoch{epoch + 1}_tmp.pth'))

        # Plotting the training and validation metrics
    epochs = range(1, cfg.train_epoch + 1)
    plt.figure(figsize=(12, 5))

    # Plotting training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig("损失.png")
    plt.legend()

    # Plotting precision, recall, and F1 score
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_precisions, 'bo-', label='Precision')
    plt.plot(epochs, val_recalls, 'ro-', label='Recall')
    plt.plot(epochs, val_f1s, 'go-', label='F1 Score')
    plt.title('Validation Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig("指标.png")
    plt.show()
    # 在训练结束后绘制所有epochs的概率分布小提琴图
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=pd.DataFrame(all_epoch_probabilities).T)  # 需要转置以符合绘图格式
    plt.title('Probability Distribution Across Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Probability')
    plt.xticks(ticks=range(cfg.train_epoch), labels=[{i + 1} for i in range(cfg.train_epoch)])
    plt.savefig("小.png")
    plt.show()
    # Convert the list of probabilities to a DataFrame and save to CSV
    probabilities_df = pd.DataFrame(all_epoch_probabilities).T
    probabilities_df.to_csv(csv_save_path, index=False)

    return model

if __name__ == '__main__':
    print(datetime.now())
    train(save=True)
    print(datetime.now())
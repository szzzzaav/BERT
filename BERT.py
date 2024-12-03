import os
import torch
import pandas as pd
import requests
import csv
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm


def load_ag_news_data(train_samples=1000, test_samples=250):  # 减少样本数量
    """
    加载AG News数据集的部分数据
    train_samples: 训练集样本数量
    test_samples: 测试集样本数量
    """
    train_path = "ag_news_train.csv"
    test_path = "ag_news_test.csv"

    # 下载文件（如果不存在）
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("下载数据集...")
        train_url = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
        test_url = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"

        for url, path in [(train_url, train_path), (test_url, test_path)]:
            if not os.path.exists(path):
                response = requests.get(url)
                with open(path, 'wb') as f:
                    f.write(response.content)

    train_data = []
    test_data = []

    # 读取部分训练数据
    with open(train_path, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        for i, row in enumerate(csv_reader):
            if i >= train_samples:  # 只读取指定数量的样本
                break
            train_data.append({
                'label': int(row[0]) - 1,
                'text': row[1] + ' ' + row[2]
            })

    # 读取部分测试数据
    with open(test_path, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        for i, row in enumerate(csv_reader):
            if i >= test_samples:  # 只读取指定数量的样本
                break
            test_data.append({
                'label': int(row[0]) - 1,
                'text': row[1] + ' ' + row[2]
            })

    return {
        'train': pd.DataFrame(train_data),
        'test': pd.DataFrame(test_data)
    }


learning_rate = 2e-5
batch_size = 16  # 减小batch size
max_length = 128
epochs = 5  # 减少训练轮数

# 初始化tokenizer和模型
print("初始化模型...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=4
)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(texts,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=max_length,
                                   return_tensors='pt')
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


def train_epoch():
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_dataloader, desc='Training')

    for batch in progress_bar:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        progress_bar.set_description(f'Loss: {loss.item():.4f}')

    return total_loss / len(train_dataloader)


def evaluate():
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    print("\n分类报告:")
    print(classification_report(true_labels, predictions,
                                target_names=['World', 'Sports', 'Business', 'Technology']))

    return predictions, true_labels


def main():
    global train_dataloader, test_dataloader, optimizer

    print(f"使用设备: {device}")

    # 加载减少规模后的数据集
    print("加载数据集...")
    dataset = load_ag_news_data(train_samples=1000, test_samples=250)

    # 准备数据
    print("准备数据集...")
    train_texts = dataset['train']['text'].tolist()
    train_labels = dataset['train']['label'].tolist()
    test_texts = dataset['test']['text'].tolist()
    test_labels = dataset['test']['label'].tolist()

    # 创建数据集实例
    train_dataset = NewsDataset(train_texts, train_labels, tokenizer, max_length)
    test_dataset = NewsDataset(test_texts, test_labels, tokenizer, max_length)

    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    # 优化器
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # 训练循环
    print("\n开始训练...")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        avg_loss = train_epoch()
        print(f"Average loss: {avg_loss:.4f}")

        # 每个epoch后评估
        print("\n评估当前epoch:")
        evaluate()


if __name__ == "__main__":
    main()
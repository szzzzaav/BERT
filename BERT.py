import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

# 1. 数据加载与预处理
print("加载数据集...")
dataset = load_dataset('ag_news')

# 初始化tokenizer和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=4
)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 训练参数
learning_rate = 2e-5
batch_size = 32
max_length = 128
epochs = 30


# 自定义数据集类
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


# 准备数据
print("准备数据集...")
train_texts = dataset['train']['text']
train_labels = dataset['train']['label']
test_texts = dataset['test']['text']
test_labels = dataset['test']['label']

# 创建数据集实例
train_dataset = NewsDataset(train_texts, train_labels, tokenizer, max_length)
test_dataset = NewsDataset(test_texts, test_labels, tokenizer, max_length)

# 创建数据加载器
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# 2. 数据分析
print("进行数据分析...")


def analyze_data():
    # 文本长度分布
    text_lengths = [len(text.split()) for text in dataset['train']['text']]
    plt.figure(figsize=(10, 6))
    plt.hist(text_lengths, bins=50)
    plt.title('Text Length Distribution')
    plt.xlabel('Number of Words')
    plt.ylabel('Count')
    plt.savefig('text_length_distribution.png')
    plt.close()


# 3. 训练模型
print("开始训练模型...")
optimizer = AdamW(model.parameters(), lr=learning_rate)


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


# 4. 评估模型
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

    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(true_labels, predictions,
                                target_names=['World', 'Sports', 'Business', 'Technology']))

    # 绘制混淆矩阵
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=['World', 'Sports', 'Business', 'Technology'],
                yticklabels=['World', 'Sports', 'Business', 'Technology'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

    return predictions, true_labels


# 5. 错误分析
def analyze_errors(texts, true_labels, predicted_labels):
    categories = ['World', 'Sports', 'Business', 'Technology']
    errors = []

    for text, true_label, pred_label in zip(texts, true_labels, predicted_labels):
        if true_label != pred_label:
            errors.append({
                'text': text,
                'true_label': categories[true_label],
                'predicted_label': categories[pred_label]
            })

    print("\n错误分析示例:")
    for error in errors[:4]:  # 展示前4个错误示例
        print(f"\n文本: {error['text'][:100]}...")
        print(f"真实标签: {error['true_label']}")
        print(f"预测标签: {error['predicted_label']}")


# 主执行流程
def main():
    print(f"使用设备: {device}")

    # 数据分析
    analyze_data()

    # 训练循环
    best_loss = float('inf')
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        avg_loss = train_epoch()
        print(f"Average loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_model.pt')

    # 加载最佳模型进行评估
    model.load_state_dict(torch.load('best_model.pt'))
    predictions, true_labels = evaluate()

    # 错误分析
    analyze_errors(test_texts, true_labels, predictions)


if __name__ == "__main__":
    main()
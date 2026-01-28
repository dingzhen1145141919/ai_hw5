import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipModel, logging
import matplotlib.pyplot as plt  # 新增：可视化库


# 关闭transformers所有警告
logging.set_verbosity_error()

# 基础配置
# 输出目录修改为：/media/ai/zcyStor/SATA/TY/hw5/models/blip_fusion/output_blip_1
OUTPUT_DIR = "/media/ai/zcyStor/SATA/TY/hw5/models/blip_fusion/output_blip_1"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # 确保目录存在

# 其他路径配置
ROOT_DIR = "/media/ai/zcyStor/SATA/TY/hw5"
DATA_DIR = os.path.join(ROOT_DIR, "data")
TRAIN_PATH = os.path.join(ROOT_DIR, "train.txt")
TEST_PATH = os.path.join(ROOT_DIR, "test_without_label.txt")
BLIP_PATH = os.path.join(ROOT_DIR, "models/blip-itm-base-coco")

# 训练参数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
EPOCHS = 10
LR = 2e-5
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

# 显存优化
torch.cuda.empty_cache()


# 兼容函数：读取不同编码的文本文件
def read_text_file(file_path):
    """兼容多种编码读取文本文件"""
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1', 'iso-8859-1']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read().strip()
        except:
            continue
    return ""


# 兼容函数：读取CSV文件（适配低版本pandas）
def read_csv_compatible(file_path):
    """兼容低版本pandas读取CSV"""
    try:
        # 先尝试utf-8
        return pd.read_csv(file_path, encoding='utf-8')
    except:
        try:
            # 再尝试gbk
            return pd.read_csv(file_path, encoding='gbk')
        except:
            # 最后尝试latin-1
            return pd.read_csv(file_path, encoding='latin-1')


# 数据集（完全兼容低版本）
class SimpleDataset(Dataset):
    def __init__(self, df, processor, is_test=False):
        self.df = df
        self.processor = processor
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        guid = str(row["guid"])

        # 加载图像
        try:
            img = Image.open(os.path.join(DATA_DIR, f"{guid}.jpg")).convert("RGB")
        except:
            # 图像读取失败时用空白图像
            img = Image.new('RGB', (224, 224), color='white')

        # 加载文本（兼容编码）
        txt_path = os.path.join(DATA_DIR, f"{guid}.txt")
        text = read_text_file(txt_path) if os.path.exists(txt_path) else ""

        # 处理空文本
        if not text:
            text = "empty"

        # 编码
        encoding = self.processor(
            images=img,
            text=text,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )

        # 标签
        try:
            label = LABEL2ID[row["tag"]] if not self.is_test else 0
        except:
            label = 1  # 默认中性

        return {
            "pixel_values": encoding["pixel_values"].squeeze(),
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long)
        }


# 模型定义
class SimpleBLIP(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载BLIP模型（忽略权重不匹配）
        self.blip = BlipModel.from_pretrained(
            BLIP_PATH,
            local_files_only=True,
            ignore_mismatched_sizes=True
        )
        # 冻结所有BLIP层
        for param in self.blip.parameters():
            param.requires_grad = False
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(768 * 2, 768),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(768, 3)
        )

    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        # 获取BLIP输出
        outputs = self.blip(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # 提取特征（兼容不同版本）
        img_feat = outputs.vision_model_output.pooler_output if hasattr(outputs.vision_model_output,
                                                                        'pooler_output') else outputs.vision_model_output.last_hidden_state[
                                                                                              :, 0, :]
        text_feat = outputs.text_model_output.pooler_output if hasattr(outputs.text_model_output,
                                                                       'pooler_output') else outputs.text_model_output.last_hidden_state[
                                                                                             :, 0, :]

        # 拼接特征
        feat = torch.cat([img_feat, text_feat], dim=-1)
        logits = self.classifier(feat)

        # 计算损失
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}


# 新增：生成训练过程的可视化图表
def plot_training_curves(train_log):
    """绘制损失、准确率、Macro-F1的变化曲线"""
    epochs = [x["epoch"] for x in train_log]
    train_loss = [x["train_loss"] for x in train_log]
    val_loss = [x["val_loss"] for x in train_log]
    val_acc = [x["val_acc"] for x in train_log]
    val_macro_f1 = [x["val_macro_f1"] for x in train_log]

    # 创建2x2的图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 训练损失 vs 验证损失
    ax1.plot(epochs, train_loss, label="训练损失", color="blue", marker="o")
    ax1.plot(epochs, val_loss, label="验证损失", color="red", marker="s")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("损失值")
    ax1.set_title("训练/验证损失曲线")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 验证准确率曲线
    ax2.plot(epochs, val_acc, label="验证准确率", color="green", marker="^")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("准确率")
    ax2.set_title("验证准确率变化曲线")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 验证Macro-F1曲线
    ax3.plot(epochs, val_macro_f1, label="验证Macro-F1", color="orange", marker="x")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Macro-F1")
    ax3.set_title("验证Macro-F1变化曲线")
    ax3.set_ylim(0, 1)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 损失+F1合并曲线
    ax4.plot(epochs, val_loss, label="验证损失", color="red", linestyle="--")
    ax4.plot(epochs, val_macro_f1, label="验证Macro-F1", color="orange")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("数值")
    ax4.set_title("验证损失 vs Macro-F1")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 保存图表到输出目录
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "training_curves_blip.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\n训练曲线已保存到：{plot_path}")


# 训练函数
def train():
    # 加载处理器
    processor = BlipProcessor.from_pretrained(
        BLIP_PATH,
        local_files_only=True,
        ignore_mismatched_sizes=True
    )

    # 加载数据（兼容低版本pandas）
    train_df = read_csv_compatible(TRAIN_PATH)
    # 清理标签列
    train_df["tag"] = train_df["tag"].fillna("neutral")
    train_df["tag"] = train_df["tag"].str.lower()

    # 划分数据集
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.2,
        stratify=train_df["tag"],
        random_state=42
    )

    # 数据加载器（最低配置）
    train_loader = DataLoader(
        SimpleDataset(train_df, processor),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    val_loader = DataLoader(
        SimpleDataset(val_df, processor),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # 模型初始化
    model = SimpleBLIP().to(DEVICE)
    optimizer = optim.AdamW(
        model.classifier.parameters(),
        lr=LR,
        eps=1e-8
    )

    best_f1 = 0.0
    train_log = []  # 新增：记录训练过程的指标，用于可视化
    print("开始训练BLIP模型...")

    for epoch in range(EPOCHS):
        print(f"\n=== Epoch {epoch + 1}/{EPOCHS} ===")

        # 训练阶段
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc="Training")

        for batch in train_bar:
            optimizer.zero_grad()
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            outputs = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["label"]
            )

            loss = outputs["loss"]
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        # 验证阶段
        model.eval()
        all_preds = []
        all_labels = []
        val_loss = 0.0
        val_bar = tqdm(val_loader, desc="Validating")

        with torch.no_grad():
            for batch in val_bar:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["label"]
                )

                val_loss += outputs["loss"].item()
                preds = torch.argmax(outputs["logits"], dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch["label"].cpu().numpy())
                val_bar.set_postfix({"loss": f"{outputs['loss'].item():.4f}"})

        # 计算指标
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        acc = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average="macro")

        # 记录当前epoch的指标（用于后续可视化）
        train_log.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_acc": acc,
            "val_macro_f1": macro_f1
        })

        # 打印过程值
        print(f"\n训练损失: {avg_train_loss:.4f} | 验证损失: {avg_val_loss:.4f}")
        print(f"验证准确率: {acc:.4f} | 验证Macro-F1: {macro_f1:.4f}")

        # 更新最佳F1
        best_f1 = max(best_f1, macro_f1)
        print(f"当前最佳F1: {best_f1:.4f}")

        # 清空显存
        torch.cuda.empty_cache()

    # 训练结束后：保存训练日志到CSV + 生成可视化图表
    log_path = os.path.join(OUTPUT_DIR, "blip_train_log.csv")
    pd.DataFrame(train_log).to_csv(log_path, index=False)
    print(f"\n训练日志已保存到：{log_path}")

    # 生成可视化图表
    plot_training_curves(train_log)

    print(f"\n训练完成！最终最佳Macro-F1: {best_f1:.4f}")
    return best_f1


# 主函数
if __name__ == "__main__":
    # 初始化
    torch.cuda.empty_cache()

    # 开始训练
    try:
        best_f1 = train()
        print(f"\n训练成功！最佳Macro-F1: {best_f1:.4f}")
    except Exception as e:
        print(f"\n训练过程中出现错误: {str(e)}")
        # 打印详细错误信息
        import traceback
        traceback.print_exc()
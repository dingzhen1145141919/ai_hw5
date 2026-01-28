import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import random
from PIL import Image, ImageEnhance
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipModel, logging
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt

# 关闭所有警告
logging.set_verbosity_error()
import warnings

warnings.filterwarnings("ignore")

# ====================== 1. 配置（修正输出目录到models/blip_fusion/output_blip_2） ======================
ROOT_DIR = "/media/ai/zcyStor/SATA/TY/hw5"
DATA_DIR = os.path.join(ROOT_DIR, "data")
TRAIN_LABEL_PATH = os.path.join(ROOT_DIR, "train.txt")
TEST_LABEL_PATH = os.path.join(ROOT_DIR, "test_without_label.txt")
BLIP_PATH = os.path.join(ROOT_DIR, "models/blip-itm-base-coco")

# 核心修正：输出目录改为 models/blip_fusion/output_blip_2
OUTPUT_DIR = os.path.join(ROOT_DIR, "models/blip_fusion/output_blip_2")
os.makedirs(OUTPUT_DIR, exist_ok=True)  # 确保目录存在，解决保存失败问题

# Training Parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
TEXT_MAX_LEN = 64
IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 30
BASE_LR = 2e-5
PRETRAIN_LR_RATIO = 0.4
WEIGHT_DECAY = 1e-4
PATIENCE = 6
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

# 严格按照你的要求配置绘图样式（仅保留DejaVu Sans英文显示）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 英文显示，避免中文乱码
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


# Set random seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


set_seed(SEED)


# ====================== 2. 数据集 ======================
class BLIPEarlyFusionDataset(Dataset):
    def __init__(self, label_df, split="train"):
        self.split = split
        self.valid_data = self._filter_valid_files(label_df)
        self.processor = BlipProcessor.from_pretrained(BLIP_PATH, local_files_only=True)

    def _filter_valid_files(self, label_df):
        valid_rows = []
        for _, row in label_df.iterrows():
            guid = str(row["guid"])
            img_path = os.path.join(DATA_DIR, f"{guid}.jpg")
            txt_path = os.path.join(DATA_DIR, f"{guid}.txt")
            if (self.split != "train" or pd.notna(row.get("tag"))) and os.path.exists(img_path) and os.path.exists(
                    txt_path):
                valid_rows.append(row)
        return pd.DataFrame(valid_rows).reset_index(drop=True)

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        row = self.valid_data.iloc[idx]
        guid = str(row["guid"])
        tag = row["tag"] if self.split != "test" else None

        # 文本处理
        txt_path = os.path.join(DATA_DIR, f"{guid}.txt")
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
        text = ""
        for encoding in encodings:
            try:
                with open(txt_path, "r", encoding=encoding, errors="ignore") as f:
                    text = f.read().strip()
                break
            except:
                continue
        if not text:
            text = "empty"

        # 文本增强
        if self.split == "train":
            if random.random() < 0.1 and len(text.split()) > 5:
                words = text.split()
                keep_idx = random.sample(range(len(words)), int(len(words) * 0.95))
                keep_idx.sort()
                words = [words[i] for i in keep_idx]
                text = " ".join(words)
            if random.random() < 0.05 and len(text.split()) > 4:
                words = text.split()
                insert_idx = random.randint(0, len(words))
                emotion_words = ["happy", "sad", "angry", "love", "hate", "great", "terrible"]
                words.insert(insert_idx, random.choice(emotion_words))
                text = " ".join(words)

        # 图像处理
        img_path = os.path.join(DATA_DIR, f"{guid}.jpg")
        image = Image.open(img_path).convert("RGB")

        if self.split == "train":
            if random.random() < 0.3:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.1:
                angle = random.randint(-15, 15)
                image = image.rotate(angle, expand=False)
            if random.random() < 0.1:
                width, height = image.size
                crop_size = int(min(width, height) * random.uniform(0.8, 1.0))
                left = random.randint(0, width - crop_size)
                top = random.randint(0, height - crop_size)
                image = image.crop((left, top, left + crop_size, top + crop_size))
                image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)

            # 图像增强
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(random.uniform(1.2, 1.8))
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(random.uniform(1.2, 2.0))
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.6, 1.4))
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(random.uniform(0.6, 1.4))

        # BLIP编码
        encoding = self.processor(
            images=image,
            text=text,
            padding="max_length",
            truncation=True,
            max_length=TEXT_MAX_LEN,
            return_tensors="pt"
        )

        # Label processing
        label = LABEL2ID[tag] if self.split != "test" else -1
        return {
            "guid": guid,
            "pixel_values": encoding["pixel_values"].squeeze(),
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long)
        }


# 数据加载器
def get_data_loaders():
    def read_csv_compatible(file_path):
        """修复read_csv的errors参数问题，兼容多编码"""
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
        for enc in encodings:
            try:
                # 移除错误的errors参数，改用on_bad_lines跳过坏行
                return pd.read_csv(file_path, encoding=enc, on_bad_lines='skip')
            except:
                continue
        # 最终兜底方案
        return pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')

    train_df = read_csv_compatible(TRAIN_LABEL_PATH)
    train_df["tag"] = train_df["tag"].fillna("neutral")
    train_df["tag"] = train_df["tag"].str.lower()

    train_df, val_df = train_test_split(
        train_df, test_size=0.2, random_state=SEED, stratify=train_df["tag"]
    )

    train_dataset = BLIPEarlyFusionDataset(train_df, split="train")
    val_dataset = BLIPEarlyFusionDataset(val_df, split="val")
    test_dataset = BLIPEarlyFusionDataset(read_csv_compatible(TEST_LABEL_PATH), split="test")

    return (
        DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False),
        DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False),
        DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    )


# ====================== 3. BLIP模型 ======================
class BLIPEarlyFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载BLIP模型
        self.blip = BlipModel.from_pretrained(
            BLIP_PATH,
            local_files_only=True,
            ignore_mismatched_sizes=True
        )

        # 模态对齐层
        self.modal_align = nn.Sequential(
            nn.Linear(768, 768),
            nn.LayerNorm(768),
            nn.ReLU()
        )

        # 图像噪声门控
        self.image_gate = nn.Sequential(
            nn.Linear(768, 1),
            nn.Sigmoid()
        )

        # 融合分类头
        self.fusion_linear1 = nn.Linear(768 * 2, 768)
        self.fusion_norm = nn.LayerNorm(768)
        self.fusion_dropout = nn.Dropout(0.4)
        self.fusion_linear2 = nn.Linear(768, 3)
        self.relu = nn.ReLU()

        # 冻结BLIP层
        self._freeze_blip_layers()

    def _freeze_blip_layers(self):
        # 先冻结所有层
        for param in self.blip.parameters():
            param.requires_grad = False

        # 解冻文本编码器最后1层
        try:
            for param in self.blip.text_model.encoder.layer[-1].parameters():
                param.requires_grad = True
        except Exception as e:
            print(f"文本编码器解冻警告: {e}")
            pass

        # 解冻视觉编码器最后1层
        try:
            for param in self.blip.vision_model.encoder.layers[-1].parameters():
                param.requires_grad = True
        except Exception as e:
            print(f"视觉编码器解冻警告: {e}")
            pass

    def forward(self, pixel_values, input_ids, attention_mask):
        # 获取BLIP的图文特征
        outputs = self.blip(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # 提取基础特征（兼容不同版本的BLIP输出）
        try:
            img_feat = outputs.vision_model_output.pooler_output
            text_feat = outputs.text_model_output.pooler_output
        except:
            img_feat = outputs.vision_model_output.last_hidden_state[:, 0, :]
            text_feat = outputs.text_model_output.last_hidden_state[:, 0, :]

        # 模态特征归一化
        text_feat = torch.nn.functional.normalize(text_feat, p=2, dim=1)
        img_feat = torch.nn.functional.normalize(img_feat, p=2, dim=1)

        # 图像特征投影到文本语义空间
        img_feat = self.modal_align(img_feat)

        # 文本指导的图像门控
        gate_weight = self.image_gate(text_feat)
        img_feat = img_feat * gate_weight

        # 特征拼接
        fused_feat = torch.cat([text_feat, img_feat], dim=1)

        # 轻量融合+残差连接
        x = self.fusion_linear1(fused_feat)
        x = self.fusion_norm(x)
        x = self.relu(x)
        x = self.fusion_dropout(x)
        x = x + text_feat  # 残差连接
        logits = self.fusion_linear2(x)

        # 伪注意力（用于日志分析）
        dummy_attn = torch.zeros((fused_feat.shape[0], 2)).to(DEVICE)
        dummy_attn[:, 0] = 1 - gate_weight.squeeze(-1)
        dummy_attn[:, 1] = gate_weight.squeeze(-1)

        return logits, dummy_attn


# ====================== 4. 训练逻辑 ======================
def calculate_metrics(all_labels, all_preds):
    return {
        "accuracy": round(accuracy_score(all_labels, all_preds), 4),
        "macro_f1": round(f1_score(all_labels, all_preds, average="macro"), 4),
        "weighted_f1": round(f1_score(all_labels, all_preds, average="weighted"), 4)
    }


def train_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Training"):
        batch = {k: v.to(DEVICE) for k, v in batch.items() if k != "guid"}
        optimizer.zero_grad()
        with autocast():
            logits, _ = model(
                batch["pixel_values"], batch["input_ids"], batch["attention_mask"]
            )
            loss = criterion(logits, batch["label"])
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_epoch(model, loader, criterion, save_bad_case=False, case_type="blip_early_fusion"):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_attn = []
    bad_cases = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            guids = batch["guid"]
            batch_device = {k: v.to(DEVICE) for k, v in batch.items() if k != "guid"}

            logits, attn_weights = model(
                batch_device["pixel_values"],
                batch_device["input_ids"],
                batch_device["attention_mask"]
            )
            loss = criterion(logits, batch_device["label"])
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            true_labels = batch_device["label"].cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(true_labels)
            all_attn.extend(attn_weights.cpu().numpy())

            if save_bad_case:
                for idx, (guid, true_id, pred_id) in enumerate(zip(guids, true_labels, preds)):
                    if true_id != pred_id:
                        bad_cases.append({
                            "guid": guid,
                            "true_label": ID2LABEL[true_id],
                            "pred_label": ID2LABEL[pred_id],
                            "true_label_id": true_id,
                            "pred_label_id": pred_id,
                            "text_weight": float(attn_weights[idx][0].cpu().numpy()),
                            "image_weight": float(attn_weights[idx][1].cpu().numpy())
                        })

    if save_bad_case and len(bad_cases) > 0:
        bad_case_df = pd.DataFrame(bad_cases)
        bad_case_path = os.path.join(OUTPUT_DIR, f"{case_type}_bad_cases.csv")
        bad_case_df.to_csv(bad_case_path, index=False, encoding="utf-8")
        print(f"\nBad cases saved to: {bad_case_path}")
        print(f"Total bad cases: {len(bad_cases)}")

    return total_loss / len(loader), calculate_metrics(all_labels, all_preds), np.mean(all_attn, axis=0)


def train_blip_early_fusion():
    train_loader, val_loader, test_loader = get_data_loaders()
    model = BLIPEarlyFusionModel().to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    # 分层学习率
    param_groups = [
        # BLIP预训练层：小学习率
        {"params": [p for p in model.blip.parameters() if p.requires_grad],
         "lr": BASE_LR * PRETRAIN_LR_RATIO, "weight_decay": WEIGHT_DECAY},
        # 模态对齐+门控：中等学习率
        {"params": model.modal_align.parameters(), "lr": BASE_LR * 0.8, "weight_decay": WEIGHT_DECAY},
        {"params": model.image_gate.parameters(), "lr": BASE_LR * 0.8, "weight_decay": WEIGHT_DECAY},
        # 融合头：大学习率
        {"params": list(model.fusion_linear1.parameters()) + list(model.fusion_linear2.parameters()),
         "lr": BASE_LR, "weight_decay": WEIGHT_DECAY}
    ]
    optimizer = optim.AdamW(param_groups)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)
    scaler = GradScaler()

    best_macro_f1 = 0.0
    patience_counter = 0
    train_log = []
    attn_log = []
    best_model_path = os.path.join(OUTPUT_DIR, "best_blip_early_fusion_model.pth")

    for epoch in range(EPOCHS):
        print(f"\n=== Epoch {epoch + 1}/{EPOCHS} (BLIP Early Fusion) ===")
        # Training
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler)
        scheduler.step()

        # Evaluation
        save_bad_case = (epoch == EPOCHS - 1 or patience_counter >= PATIENCE - 1)
        val_loss, val_metrics, avg_attn = eval_epoch(model, val_loader, criterion, save_bad_case=save_bad_case)

        # Logging
        train_log.append({
            "epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, **val_metrics
        })
        attn_log.append({
            "epoch": epoch + 1, "text_weight": avg_attn[0], "image_weight": avg_attn[1]
        })

        # Print results
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Acc: {val_metrics['accuracy']:.4f} | Val Macro-F1: {val_metrics['macro_f1']:.4f}")
        print(f"Average Attention Weights (Text/Image): {avg_attn[0]:.4f} / {avg_attn[1]:.4f}")

        # Early stopping
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            # 修复：只保存模型权重，不保存其他对象
            torch.save({
                "model_state_dict": model.state_dict(),
                "best_metrics": val_metrics,
                "epoch": epoch + 1
            }, best_model_path)
            patience_counter = 0
            print(f"Update Best BLIP Model! F1: {best_macro_f1:.4f} (Epoch {epoch + 1})")
        else:
            patience_counter += 1
            print(f"F1 No Improvement! Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                eval_epoch(model, val_loader, criterion, save_bad_case=True)
                print(f"Early Stopping Triggered! Best BLIP F1: {best_macro_f1:.4f}")
                break

    # 修复：移除weights_only=True，解决反序列化错误
    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Load Best BLIP Model (Epoch {checkpoint['epoch']}, F1 {best_macro_f1:.4f})")

    # Save logs（单独保存日志，不放在模型checkpoint中）
    pd.DataFrame(train_log).to_csv(os.path.join(OUTPUT_DIR, "blip_early_fusion_train_log.csv"), index=False)
    pd.DataFrame(attn_log).to_csv(os.path.join(OUTPUT_DIR, "blip_attention_log.csv"), index=False)

    return best_macro_f1, train_log, attn_log, model, test_loader


# 测试集预测（修复read_csv的errors参数问题）
def predict_test_set(best_model, test_loader):
    model = best_model.to(DEVICE)
    model.eval()

    all_guids = []
    all_preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting Test Set"):
            guids = batch["guid"]
            batch_device = {k: v.to(DEVICE) for k, v in batch.items() if k != "guid"}
            logits, _ = model(
                batch_device["pixel_values"],
                batch_device["input_ids"],
                batch_device["attention_mask"]
            )
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_guids.extend(guids)
            all_preds.extend([ID2LABEL[p] for p in preds])

    output_df = pd.DataFrame({"guid": all_guids, "tag": all_preds})

    # 核心修复：移除read_csv的errors参数，改用兼容的读取方式
    def read_test_csv(file_path):
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
        for enc in encodings:
            try:
                return pd.read_csv(file_path, encoding=enc, on_bad_lines='skip')
            except:
                continue
        return pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')

    original_test_df = read_test_csv(TEST_LABEL_PATH)
    # 修复guid类型问题
    original_test_df["guid"] = original_test_df["guid"].astype(str)
    output_df["guid"] = output_df["guid"].astype(str)
    output_df = output_df.merge(original_test_df[["guid"]], on="guid", how="right")
    output_df.to_csv(os.path.join(OUTPUT_DIR, "blip_test_result.txt"), index=False, header=False)
    print(f"\nTest Set Prediction Completed! Results saved to: {os.path.join(OUTPUT_DIR, 'blip_test_result.txt')}")


# 可视化（严格匹配你的字体配置，修复画图逻辑）
def plot_training_curves(fusion_log, attn_log):
    try:
        # 提取绘图数据
        epochs = [x["epoch"] for x in fusion_log]
        fusion_train_loss = [x["train_loss"] for x in fusion_log]
        fusion_val_loss = [x["val_loss"] for x in fusion_log]
        fusion_macro_f1 = [x["macro_f1"] for x in fusion_log]
        fusion_acc = [x["accuracy"] for x in fusion_log]

        attn_epochs = [x["epoch"] for x in attn_log]
        text_weight = [x["text_weight"] for x in attn_log]
        image_weight = [x["image_weight"] for x in attn_log]

        # 确保数据长度一致
        min_epoch_len = min(len(epochs), len(attn_epochs))
        epochs = epochs[:min_epoch_len]
        fusion_train_loss = fusion_train_loss[:min_epoch_len]
        fusion_val_loss = fusion_val_loss[:min_epoch_len]
        fusion_macro_f1 = fusion_macro_f1[:min_epoch_len]
        fusion_acc = fusion_acc[:min_epoch_len]
        attn_epochs = attn_epochs[:min_epoch_len]
        text_weight = text_weight[:min_epoch_len]
        image_weight = image_weight[:min_epoch_len]

        # 创建2x2子图（严格匹配你的样式）
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 子图1：训练/验证损失曲线
        ax1.plot(epochs, fusion_train_loss, label="Training Loss", color="blue", marker="o", linewidth=2)
        ax1.plot(epochs, fusion_val_loss, label="Validation Loss", color="red", marker="s", linewidth=2)
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Loss Value", fontsize=12)
        ax1.set_title("BLIP Early Fusion Training/Validation Loss Curves", fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)

        # 子图2：Macro-F1曲线
        ax2.plot(epochs, fusion_macro_f1, label="BLIP Early Fusion Macro-F1", color="green", marker="^", linewidth=2)
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("Macro-F1 Value", fontsize=12)
        ax2.set_ylim(0, 1)  # F1值范围0-1
        ax2.set_title("BLIP Early Fusion Validation Macro-F1 Curve", fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)

        # 子图3：文本/图像注意力权重曲线
        ax3.plot(attn_epochs, text_weight, label="Text Weight", color="purple", marker="o", linewidth=2)
        ax3.plot(attn_epochs, image_weight, label="Image Weight", color="orange", marker="s", linewidth=2)
        ax3.set_xlabel("Epoch", fontsize=12)
        ax3.set_ylabel("Attention Weight (0-1)", fontsize=12)
        ax3.set_title("Text/Image Dynamic Weight (BLIP Early Fusion)", fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=10)

        # 子图4：准确率曲线
        ax4.plot(epochs, fusion_acc, label="BLIP Early Fusion Accuracy", color="darkblue", marker="x", linewidth=2)
        ax4.set_xlabel("Epoch", fontsize=12)
        ax4.set_ylabel("Accuracy Value", fontsize=12)
        ax4.set_ylim(0, 1)  # 准确率范围0-1
        ax4.set_title("BLIP Early Fusion Validation Accuracy Curve", fontsize=14)
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=10)

        # 调整布局并保存（严格匹配你的DPI配置）
        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, "blip_training_curves.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\nTraining curves saved to: {save_path}")

        # 释放plt资源
        plt.close(fig)

    except Exception as e:
        print(f"\n⚠️ 可视化绘图失败: {e}")
        print("提示：训练日志已保存，可使用独立的CSV可视化脚本重新绘图")


# ====================== 主函数 ======================
if __name__ == "__main__":
    # 清空显存
    torch.cuda.empty_cache()

    # 训练BLIP Early Fusion模型
    try:
        best_f1, train_log, attn_log, best_model, test_loader = train_blip_early_fusion()

        # 预测测试集
        predict_test_set(best_model, test_loader)

        # 可视化（严格匹配你的字体配置）
        plot_training_curves(train_log, attn_log)

        print("\n===== Final Results Summary (BLIP Early Fusion Model) =====")
        print(f"BLIP Early Fusion Model Best Macro-F1: {best_f1:.4f}")
        print(f"\nAll output files are saved to: {OUTPUT_DIR}")
        print(f"\nKey output files:")
        print("- blip_early_fusion_train_log.csv (训练日志)")
        print("- blip_attention_log.csv (图文权重日志)")
        print("- blip_early_fusion_bad_cases.csv (错误案例分析)")
        print("- blip_test_result.txt (测试集预测结果)")
        print("- blip_training_curves.png (训练曲线可视化)")
        print("- best_blip_early_fusion_model.pth (最优模型权重)")
    except Exception as e:
        print(f"\n运行错误: {str(e)}")
        import traceback

        traceback.print_exc()
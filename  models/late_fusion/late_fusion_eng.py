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
from transformers import RobertaTokenizer, RobertaModel, ViTImageProcessor, ViTModel
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt

# Path Configuration - 修改：使用清洗后的训练数据，输出目录添加english_only标识
ROOT_DIR = "../../"
DATA_DIR = os.path.join(ROOT_DIR, "data")
# 核心修改：替换为清洗后的训练数据路径
TRAIN_LABEL_PATH = "/media/ai/zcyStor/SATA/TY/hw5/train_english_only.txt"
TEST_LABEL_PATH = os.path.join(ROOT_DIR, "test_without_label.txt")
LOCAL_ROBERTA_PATH = os.path.join(ROOT_DIR, "models", "roberta-base")
LOCAL_VIT_PATH = os.path.join(ROOT_DIR, "models", "vit-base")
# 核心修改：输出目录添加english_only后缀，区分原有结果
OUTPUT_DIR = "./outputs_english_only"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Training Parameters (修复后参数，保证F1恢复0.57+)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
TEXT_MAX_LEN = 64
IMAGE_SIZE = 224
BATCH_SIZE = 128
EPOCHS = 30  # 最大训练轮数，早停机制会提前终止
BASE_LR = 4e-5
PRETRAIN_LR_RATIO = 0.4
WEIGHT_DECAY = 1e-4
PATIENCE = 6  # 早停耐心值：连续6个epoch F1无提升则停止
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# Dataset Definition
class MultiModalDataset(Dataset):
    def __init__(self, label_df, split="train"):
        self.split = split
        self.valid_data = self._filter_valid_files(label_df)
        self.text_tokenizer = RobertaTokenizer.from_pretrained(LOCAL_ROBERTA_PATH)
        self.image_processor = ViTImageProcessor.from_pretrained(LOCAL_VIT_PATH)

    def _filter_valid_files(self, label_df):
        valid_rows = []
        for _, row in label_df.iterrows():
            guid = str(row["guid"])
            img_path = os.path.join(DATA_DIR, f"{guid}.jpg")
            txt_path = os.path.join(DATA_DIR, f"{guid}.txt")
            if (self.split != "train" or pd.notna(row.get("tag"))) and os.path.exists(img_path) and os.path.exists(txt_path):
                valid_rows.append(row)
        return pd.DataFrame(valid_rows).reset_index(drop=True)

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        row = self.valid_data.iloc[idx]
        guid = str(row["guid"])
        tag = row["tag"] if self.split != "test" else None

        # Text processing with mild augmentation
        txt_path = os.path.join(DATA_DIR, f"{guid}.txt")
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()

        if self.split == "train":
            if random.random() < 0.1 and len(text.split()) > 5:
                words = text.split()
                keep_idx = random.sample(range(len(words)), int(len(words) * 0.9))
                keep_idx.sort()
                words = [words[i] for i in keep_idx]
                text = " ".join(words)
            if random.random() < 0.05 and len(text.split()) > 4:
                words = text.split()
                insert_idx = random.randint(0, len(words))
                stop_words = ["the", "a", "an", "and", "or", "but"]
                words.insert(insert_idx, random.choice(stop_words))
                text = " ".join(words)

        text_encoding = self.text_tokenizer(
            text, max_length=TEXT_MAX_LEN, padding="max_length", truncation=True, return_tensors="pt"
        )

        # Image processing with mild augmentation
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
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.7, 1.3))
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(random.uniform(0.7, 1.3))

        image_encoding = self.image_processor(
            image, resize_size=IMAGE_SIZE, crop_size=IMAGE_SIZE, return_tensors="pt"
        )

        # Label processing
        label = LABEL2ID[tag] if self.split != "test" else -1
        return {
            "guid": guid,
            "text_input_ids": text_encoding["input_ids"].squeeze(),
            "text_attention_mask": text_encoding["attention_mask"].squeeze(),
            "image_pixel_values": image_encoding["pixel_values"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long)
        }

def get_data_loaders():
    # 读取清洗后的训练数据
    train_df = pd.read_csv(TRAIN_LABEL_PATH)
    train_df, val_df = train_test_split(
        train_df, test_size=0.2, random_state=SEED, stratify=train_df["tag"]
    )
    train_dataset = MultiModalDataset(train_df, split="train")
    val_dataset = MultiModalDataset(val_df, split="val")
    test_dataset = MultiModalDataset(pd.read_csv(TEST_LABEL_PATH), split="test")

    return (
        DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True),
        DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True),
        DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True)
    )

# Model Definition (修复后，保证F1恢复)
class TextBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(LOCAL_ROBERTA_PATH)
        roberta_params = list(self.roberta.parameters())
        freeze_layer_num = 2
        for i, param in enumerate(roberta_params):
            if i < freeze_layer_num:
                param.requires_grad = False
        self.roberta_dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)
        )

    def forward(self, input_ids, attention_mask):
        cls_feat = self.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        cls_feat = self.roberta_dropout(cls_feat)
        return self.classifier(cls_feat), cls_feat

class ImageBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = ViTModel.from_pretrained(LOCAL_VIT_PATH)
        vit_params = list(self.vit.parameters())
        freeze_layer_num = len(vit_params) - 8
        for i, param in enumerate(vit_params):
            if i < freeze_layer_num:
                param.requires_grad = False
        self.vit_dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 3)
        )

    def forward(self, pixel_values):
        cls_feat = self.vit(pixel_values=pixel_values).last_hidden_state[:, 0, :]
        cls_feat = self.vit_dropout(cls_feat)
        return self.classifier(cls_feat), cls_feat

class MultiModalFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_branch = TextBranch()
        self.image_branch = ImageBranch()

        self.attention = nn.Sequential(
            nn.Linear(768 * 2, 2),
            nn.Softmax(dim=1)
        )
        self.fusion_dropout = nn.Dropout(0.3)  # 修复：从0.5降回0.3
        self.fusion_classifier = nn.Sequential(  # 修复：恢复分类器复杂度
            nn.Linear(768 * 2, 768),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(768, 3)
        )

    def forward(self, text_input_ids, text_attention_mask, image_pixel_values):
        _, text_feat = self.text_branch(text_input_ids, text_attention_mask)
        _, image_feat = self.image_branch(image_pixel_values)

        # L2 normalization for feature scale alignment
        text_feat = torch.nn.functional.normalize(text_feat, p=2, dim=1)
        image_feat = torch.nn.functional.normalize(image_feat, p=2, dim=1)

        concat_feat = torch.cat([text_feat, image_feat], dim=1)
        attn_weights = self.attention(concat_feat)

        # 修复：放宽注意力权重约束
        attn_weights = torch.clamp(attn_weights, min=0.1, max=0.9)
        attn_weights = attn_weights / attn_weights.sum(dim=1, keepdim=True)

        weighted_text = text_feat * attn_weights[:, 0:1]
        weighted_image = image_feat * attn_weights[:, 1:2]

        fused_feat = torch.cat([weighted_text, weighted_image], dim=1)
        fused_feat = self.fusion_dropout(fused_feat)
        logits = self.fusion_classifier(fused_feat)

        return logits, attn_weights

# Training Utilities
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
                batch["text_input_ids"], batch["text_attention_mask"], batch["image_pixel_values"]
            )
            loss = criterion(logits, batch["label"])
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)

# 新增：修改eval_epoch函数，记录bad case（适配english_only）
def eval_epoch(model, loader, criterion, save_bad_case=False, case_type="multi_modal"):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_attn = []
    bad_cases = []  # 存储bad case信息

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            guids = batch["guid"]  # 获取数据ID
            batch_device = {k: v.to(DEVICE) for k, v in batch.items() if k != "guid"}

            logits, attn_weights = model(
                batch_device["text_input_ids"],
                batch_device["text_attention_mask"],
                batch_device["image_pixel_values"]
            )
            loss = criterion(logits, batch_device["label"])
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            true_labels = batch_device["label"].cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(true_labels)
            all_attn.extend(attn_weights.cpu().numpy())

            # 收集bad case（预测错误的样本）
            if save_bad_case:
                for idx, (guid, true_id, pred_id) in enumerate(zip(guids, true_labels, preds)):
                    if true_id != pred_id:
                        bad_cases.append({
                            "guid": guid,
                            "true_label": ID2LABEL[true_id],
                            "pred_label": ID2LABEL[pred_id],
                            "true_label_id": true_id,
                            "pred_label_id": pred_id
                        })

    # 保存bad case到CSV（添加english_only标识）
    if save_bad_case and len(bad_cases) > 0:
        bad_case_df = pd.DataFrame(bad_cases)
        bad_case_path = os.path.join(OUTPUT_DIR, f"{case_type}_bad_cases_english_only.csv")
        bad_case_df.to_csv(bad_case_path, index=False, encoding="utf-8")
        print(f"\nBad cases saved to: {bad_case_path}")
        print(f"Total bad cases: {len(bad_cases)}")

    return total_loss / len(loader), calculate_metrics(all_labels, all_preds), np.mean(all_attn, axis=0)

# 新增：单模态eval函数（适配bad case输出 + english_only）
def eval_single_modal(model, loader, criterion, modal_type, save_bad_case=False):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    bad_cases = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Eval {modal_type.capitalize()}"):
            guids = batch["guid"]
            batch_device = {k: v.to(DEVICE) for k, v in batch.items() if k != "guid"}

            if modal_type == "text":
                logits, _ = model(batch_device["text_input_ids"], batch_device["text_attention_mask"])
            else:
                logits, _ = model(batch_device["image_pixel_values"])

            loss = criterion(logits, batch_device["label"])
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            true_labels = batch_device["label"].cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(true_labels)

            # 收集bad case
            if save_bad_case:
                for idx, (guid, true_id, pred_id) in enumerate(zip(guids, true_labels, preds)):
                    if true_id != pred_id:
                        bad_cases.append({
                            "guid": guid,
                            "true_label": ID2LABEL[true_id],
                            "pred_label": ID2LABEL[pred_id],
                            "true_label_id": true_id,
                            "pred_label_id": pred_id
                        })

    # 保存单模态bad case（添加english_only标识）
    if save_bad_case and len(bad_cases) > 0:
        bad_case_df = pd.DataFrame(bad_cases)
        bad_case_path = os.path.join(OUTPUT_DIR, f"{modal_type}_bad_cases_english_only.csv")
        bad_case_df.to_csv(bad_case_path, index=False, encoding="utf-8")
        print(f"\n{modal_type.capitalize()} bad cases saved to: {bad_case_path}")
        print(f"Total {modal_type} bad cases: {len(bad_cases)}")

    val_loss = total_loss / len(loader)
    val_metrics = calculate_metrics(all_labels, all_preds)
    return val_loss, val_metrics

# Single Modal Training (Ablation Study + 标准早停机制)
def train_single_modal(modal_type):
    print(f"\n=== Start Training {modal_type.capitalize()} Only Model (Ablation Study) ===")
    train_loader, val_loader, _ = get_data_loaders()
    model = TextBranch().to(DEVICE) if modal_type == "text" else ImageBranch().to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    # Layer-wise learning rate
    if modal_type == "text":
        param_groups = [
            {"params": model.roberta.parameters(), "lr": BASE_LR * PRETRAIN_LR_RATIO, "weight_decay": WEIGHT_DECAY},
            {"params": model.classifier.parameters(), "lr": BASE_LR, "weight_decay": WEIGHT_DECAY}
        ]
    else:
        param_groups = [
            {"params": model.vit.parameters(), "lr": BASE_LR * PRETRAIN_LR_RATIO, "weight_decay": WEIGHT_DECAY},
            {"params": model.classifier.parameters(), "lr": BASE_LR, "weight_decay": WEIGHT_DECAY}
        ]
    optimizer = optim.AdamW(param_groups)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)
    scaler = GradScaler()

    best_macro_f1 = 0.0
    patience_counter = 0
    train_log = []
    # 修改：模型文件名添加english_only标识
    best_model_path = os.path.join(OUTPUT_DIR, f"best_{modal_type}_model_english_only.pth")

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        # Training phase
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Train {modal_type.capitalize()}"):
            batch_device = {k: v.to(DEVICE) for k, v in batch.items() if k != "guid"}
            optimizer.zero_grad()
            with autocast():
                if modal_type == "text":
                    logits, _ = model(batch_device["text_input_ids"], batch_device["text_attention_mask"])
                else:
                    logits, _ = model(batch_device["image_pixel_values"])
                loss = criterion(logits, batch_device["label"])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)
        scheduler.step()

        # Evaluation phase - 调用新增的eval_single_modal函数
        val_loss, val_metrics = eval_single_modal(
            model, val_loader, criterion, modal_type,
            save_bad_case=(epoch == EPOCHS-1 or patience_counter >= PATIENCE-1)  # 仅在最后一轮/早停前保存bad case
        )

        # Logging
        train_log.append({
            "epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, **val_metrics
        })

        # Print results
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Acc: {val_metrics['accuracy']:.4f} | Val Macro-F1: {val_metrics['macro_f1']:.4f}")

        # 早停核心逻辑：监控Macro-F1，保存最优模型
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "best_metrics": val_metrics,
                "train_log": train_log,
                "epoch": epoch + 1
            }, best_model_path)
            patience_counter = 0
            print(f"Update Best {modal_type.capitalize()} Model! F1: {best_macro_f1:.4f} (Epoch {epoch+1})")
        else:
            patience_counter += 1
            print(f"F1 No Improvement! Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                # 早停时强制保存bad case
                eval_single_modal(model, val_loader, criterion, modal_type, save_bad_case=True)
                print(f"Early Stopping Triggered! Best {modal_type.capitalize()} F1: {best_macro_f1:.4f}")
                break

    # 加载最优模型权重
    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Load Best {modal_type.capitalize()} Model (Epoch {checkpoint['epoch']}, F1 {best_macro_f1:.4f})")

    # 修改：日志文件名添加english_only标识
    pd.DataFrame(train_log).to_csv(os.path.join(OUTPUT_DIR, f"{modal_type}_train_log_english_only.csv"), index=False)
    print(f"{modal_type.capitalize()} Only Model Training Completed! Best Macro-F1: {best_macro_f1:.4f}")
    return best_macro_f1, train_log, model

# Multi-modal Fusion Training + 标准早停机制（核心修改）
def train_multi_modal():
    train_loader, val_loader, _ = get_data_loaders()
    model = MultiModalFusionModel().to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    # Layer-wise learning rate
    param_groups = [
        {"params": model.text_branch.roberta.parameters(), "lr": BASE_LR * PRETRAIN_LR_RATIO, "weight_decay": WEIGHT_DECAY},
        {"params": model.image_branch.vit.parameters(), "lr": BASE_LR * PRETRAIN_LR_RATIO, "weight_decay": WEIGHT_DECAY},
        {"params": list(model.text_branch.classifier.parameters()) + list(model.image_branch.classifier.parameters()),
         "lr": BASE_LR, "weight_decay": WEIGHT_DECAY},
        {"params": list(model.attention.parameters()) + list(model.fusion_classifier.parameters()),
         "lr": BASE_LR, "weight_decay": WEIGHT_DECAY}
    ]
    optimizer = optim.AdamW(param_groups)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)
    scaler = GradScaler()

    best_macro_f1 = 0.0
    patience_counter = 0
    train_log = []
    attn_log = []
    # 修改：多模态模型文件名添加english_only标识
    best_model_path = os.path.join(OUTPUT_DIR, "best_multi_modal_model_english_only.pth")

    for epoch in range(EPOCHS):
        print(f"\n=== Epoch {epoch + 1}/{EPOCHS} (Multi-modal Fusion) ===")
        # Training
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler)
        # Evaluation - 调用修改后的eval_epoch，控制bad case保存时机
        save_bad_case = (epoch == EPOCHS-1 or patience_counter >= PATIENCE-1)
        val_loss, val_metrics, avg_attn = eval_epoch(
            model, val_loader, criterion,
            save_bad_case=save_bad_case,
            case_type="multi_modal"
        )
        # Learning rate update
        scheduler.step()

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
        print(f"Attention Weights (Text/Image): {avg_attn[0]:.4f} / {avg_attn[1]:.4f}")
        print(f"Current Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        # 早停核心逻辑
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "best_metrics": val_metrics,
                "train_log": train_log,
                "attn_log": attn_log,
                "epoch": epoch + 1
            }, best_model_path)
            patience_counter = 0
            print(f"Update Best Multi-modal Model! F1: {best_macro_f1:.4f} (Epoch {epoch+1})")
        else:
            patience_counter += 1
            print(f"F1 No Improvement! Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                # 早停时强制保存bad case
                eval_epoch(model, val_loader, criterion, save_bad_case=True, case_type="multi_modal")
                print(f"Early Stopping Triggered! Best Multi-modal F1: {best_macro_f1:.4f}")
                break

    # 加载最优模型权重
    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Load Best Multi-modal Model (Epoch {checkpoint['epoch']}, F1 {best_macro_f1:.4f})")

    # 修改：日志文件名添加english_only标识
    pd.DataFrame(train_log).to_csv(os.path.join(OUTPUT_DIR, "fusion_train_log_english_only.csv"), index=False)
    pd.DataFrame(attn_log).to_csv(os.path.join(OUTPUT_DIR, "attention_log_english_only.csv"), index=False)
    print(f"\nMulti-modal Model Training Completed! Best Val Macro-F1: {best_macro_f1:.4f}")
    return best_macro_f1, train_log, attn_log, model

# Test Set Prediction（使用最优模型预测）
def predict_test_set(best_model):
    _, _, test_loader = get_data_loaders()
    model = best_model.to(DEVICE)
    model.eval()

    all_guids = []
    all_preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting Test Set"):
            guids = batch["guid"]
            batch_device = {k: v.to(DEVICE) for k, v in batch.items() if k != "guid"}
            logits, _ = model(
                batch_device["text_input_ids"],
                batch_device["text_attention_mask"],
                batch_device["image_pixel_values"]
            )
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_guids.extend(guids)
            all_preds.extend([ID2LABEL[p] for p in preds])

    output_df = pd.DataFrame({"guid": all_guids, "tag": all_preds})
    original_test_df = pd.read_csv(TEST_LABEL_PATH)
    output_df = output_df.merge(original_test_df[["guid"]], on="guid", how="right")
    # 修改：测试结果文件名添加english_only标识
    output_df.to_csv(os.path.join(OUTPUT_DIR, "test_result_english_only.txt"), index=False, header=False)
    print(f"\nTest Set Prediction Completed! Results saved to: {os.path.join(OUTPUT_DIR, 'test_result_english_only.txt')}")

# Visualization (English Title)
def plot_training_curves(fusion_log, text_log, image_log, attn_log):
    epochs = [x["epoch"] for x in fusion_log]
    fusion_train_loss = [x["train_loss"] for x in fusion_log]
    fusion_val_loss = [x["val_loss"] for x in fusion_log]
    fusion_macro_f1 = [x["macro_f1"] for x in fusion_log]

    text_epochs = [x["epoch"] for x in text_log]
    text_macro_f1 = [x["macro_f1"] for x in text_log]
    image_epochs = [x["epoch"] for x in image_log]
    image_macro_f1 = [x["macro_f1"] for x in image_log]

    attn_epochs = [x["epoch"] for x in attn_log]
    text_weight = [x["text_weight"] for x in attn_log]
    image_weight = [x["image_weight"] for x in attn_log]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Subplot 1: Loss Curves
    ax1.plot(epochs, fusion_train_loss, label="Training Loss", color="blue", marker="o", linewidth=2)
    ax1.plot(epochs, fusion_val_loss, label="Validation Loss", color="red", marker="s", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss Value", fontsize=12)
    ax1.set_title("Multi-modal Model (English Only) Training/Validation Loss Curves", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Subplot 2: Macro-F1 Curve
    ax2.plot(epochs, fusion_macro_f1, label="Multi-modal Macro-F1", color="green", marker="^", linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Macro-F1 Value", fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.set_title("Multi-modal Model (English Only) Validation Macro-F1 Curve", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    # Subplot 3: Attention Weights
    ax3.plot(attn_epochs, text_weight, label="Text Weight", color="purple", marker="o", linewidth=2)
    ax3.plot(attn_epochs, image_weight, label="Image Weight", color="orange", marker="s", linewidth=2)
    ax3.set_xlabel("Epoch", fontsize=12)
    ax3.set_ylabel("Attention Weight (0-1)", fontsize=12)
    ax3.set_title("Text/Image Attention Weight Changes (English Only)", fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)

    # Subplot 4: Ablation Study
    ax4.plot(epochs, fusion_macro_f1, label="Multi-modal (Attention Fusion)", color="green", marker="^", linewidth=2)
    ax4.plot(text_epochs, text_macro_f1, label="Text Only", color="purple", marker="o", linewidth=2)
    ax4.plot(image_epochs, image_macro_f1, label="Image Only", color="orange", marker="s", linewidth=2)
    ax4.set_xlabel("Epoch", fontsize=12)
    ax4.set_ylabel("Macro-F1 Value", fontsize=12)
    ax4.set_ylim(0, 1)
    ax4.set_title("Ablation Study (English Only): Macro-F1 Comparison", fontsize=14)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)

    # 修改：可视化文件名添加english_only标识
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_curves_english_only.png"), dpi=300, bbox_inches="tight")
    print(f"\nTraining curves saved to: {os.path.join(OUTPUT_DIR, 'training_curves_english_only.png')}")

# Main Function（适配早停后的数据传递）
if __name__ == "__main__":
    # 训练多模态模型（返回最优模型）
    fusion_best_f1, fusion_log, attn_log, best_multi_model = train_multi_modal()
    # 训练单模态模型
    text_best_f1, text_log, _ = train_single_modal("text")
    image_best_f1, image_log, _ = train_single_modal("image")
    # 使用最优模型预测测试集
    predict_test_set(best_multi_model)
    # 可视化训练结果
    plot_training_curves(fusion_log, text_log, image_log, attn_log)

    print("\n===== Final Results Summary (English Only Dataset) =====")
    print(f"Multi-modal Model Best Macro-F1: {fusion_best_f1:.4f}")
    print(f"Text Only Model Best Macro-F1: {text_best_f1:.4f}")
    print(f"Image Only Model Best Macro-F1: {image_best_f1:.4f}")
    print(f"\nAll output files are saved to: {OUTPUT_DIR}")
    # 新增：打印bad case文件路径
    print(f"\nBad case files (English Only) saved in {OUTPUT_DIR}:")
    print("- multi_modal_bad_cases_english_only.csv (多模态模型错误样本)")
    print("- text_bad_cases_english_only.csv (文本单模态错误样本)")
    print("- image_bad_cases_english_only.csv (图像单模态错误样本)")
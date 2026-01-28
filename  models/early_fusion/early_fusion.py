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
# 适配低版本PyTorch：使用旧的cuda.amp接口
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt

# Path Configuration
ROOT_DIR = "../../"
DATA_DIR = os.path.join(ROOT_DIR, "data")
TRAIN_LABEL_PATH = os.path.join(ROOT_DIR, "train.txt")
TEST_LABEL_PATH = os.path.join(ROOT_DIR, "test_without_label.txt")
LOCAL_ROBERTA_PATH = os.path.join(ROOT_DIR, "models", "roberta-base")
LOCAL_VIT_PATH = os.path.join(ROOT_DIR, "models", "vit-base")
OUTPUT_DIR = "./outputs_early_fusion_optimized"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Training Parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
TEXT_MAX_LEN = 64
IMAGE_SIZE = 224
BATCH_SIZE = 128
EPOCHS = 30
BASE_LR = 4e-5
PRETRAIN_LR_RATIO = 0.4
WEIGHT_DECAY = 1e-4
PATIENCE = 6
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

        # Text processing with mild augmentation
        txt_path = os.path.join(DATA_DIR, f"{guid}.txt")
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()

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

        text_encoding = self.text_tokenizer(
            text, max_length=TEXT_MAX_LEN, padding="max_length", truncation=True, return_tensors="pt"
        )

        # Image processing with enhanced augmentation
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

            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(random.uniform(1.2, 1.8))
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(random.uniform(1.2, 2.0))
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.6, 1.4))
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(random.uniform(0.6, 1.4))

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


# Model Definition
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
        self.feature_proj = nn.Linear(768, 768)

    def forward(self, input_ids, attention_mask):
        cls_feat = self.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        cls_feat = self.roberta_dropout(cls_feat)
        cls_feat = self.feature_proj(cls_feat)
        return cls_feat


class ImageBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = ViTModel.from_pretrained(LOCAL_VIT_PATH)
        self.visual_attention = nn.Sequential(
            nn.Linear(768, 768),
            nn.Tanh(),
            nn.Linear(768, 1),
            nn.Softmax(dim=1)
        )
        vit_params = list(self.vit.parameters())
        freeze_layer_num = len(vit_params) - 8
        for i, param in enumerate(vit_params):
            if i < freeze_layer_num:
                param.requires_grad = False
        self.vit_dropout = nn.Dropout(0.2)
        self.feature_proj = nn.Linear(768, 768)

    def forward(self, pixel_values):
        vit_output = self.vit(pixel_values=pixel_values).last_hidden_state
        attn_weights = self.visual_attention(vit_output)
        weighted_feat = torch.sum(vit_output * attn_weights, dim=1)
        cls_feat = self.vit_dropout(weighted_feat)
        cls_feat = self.feature_proj(cls_feat)
        return cls_feat


class EarlyFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_branch = TextBranch()
        self.image_branch = ImageBranch()

        # 1. 模态特征对齐层
        self.modal_align = nn.Sequential(
            nn.Linear(768, 768),
            nn.LayerNorm(768),
            nn.ReLU()
        )
        # 2. 图像噪声门控
        self.image_gate = nn.Sequential(
            nn.Linear(768, 1),
            nn.Sigmoid()
        )
        # 3. 轻量融合分类头
        self.fusion_linear1 = nn.Linear(768 * 2, 768)
        self.fusion_norm = nn.LayerNorm(768)
        self.fusion_dropout = nn.Dropout(0.4)
        self.fusion_linear2 = nn.Linear(768, 3)
        self.relu = nn.ReLU()

    def forward(self, text_input_ids, text_attention_mask, image_pixel_values):
        # 1. 提取文本/图像特征
        text_feat = self.text_branch(text_input_ids, text_attention_mask)
        image_feat = self.image_branch(image_pixel_values)

        # 2. 模态特征归一化
        text_feat = torch.nn.functional.normalize(text_feat, p=2, dim=1)
        image_feat = torch.nn.functional.normalize(image_feat, p=2, dim=1)

        # 3. 图像特征投影到文本语义空间
        image_feat = self.modal_align(image_feat)

        # 4. 文本指导的图像门控
        gate_weight = self.image_gate(text_feat)
        image_feat = image_feat * gate_weight

        # 5. 特征拼接
        fused_feat = torch.cat([text_feat, image_feat], dim=1)

        # 6. 轻量融合+残差连接
        x = self.fusion_linear1(fused_feat)
        x = self.fusion_norm(x)
        x = self.relu(x)
        x = self.fusion_dropout(x)
        x = x + text_feat
        logits = self.fusion_linear2(x)

        # 伪注意力
        dummy_attn = torch.zeros((fused_feat.shape[0], 2)).to(DEVICE)
        dummy_attn[:, 0] = 1 - gate_weight.squeeze(-1)
        dummy_attn[:, 1] = gate_weight.squeeze(-1)
        return logits, dummy_attn


# Training Utilities
def calculate_metrics(all_labels, all_preds):
    """修复核心错误：显式指定多分类的average参数"""
    return {
        "accuracy": round(accuracy_score(all_labels, all_preds), 4),
        "macro_f1": round(f1_score(all_labels, all_preds, average="macro"), 4),  # 关键修复
        "weighted_f1": round(f1_score(all_labels, all_preds, average="weighted"), 4)
    }


def train_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Training"):
        batch = {k: v.to(DEVICE) for k, v in batch.items() if k != "guid"}
        optimizer.zero_grad()
        # 适配低版本PyTorch：移除device_type参数
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


def eval_epoch(model, loader, criterion, save_bad_case=False, case_type="early_fusion"):
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
        bad_case_path = os.path.join(OUTPUT_DIR, f"{case_type}_bad_cases_early_fusion.csv")
        bad_case_df.to_csv(bad_case_path, index=False, encoding="utf-8")
        print(f"\nBad cases saved to: {bad_case_path}")
        print(f"Total bad cases: {len(bad_cases)}")

    return total_loss / len(loader), calculate_metrics(all_labels, all_preds), np.mean(all_attn, axis=0)


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
                feat = model(batch_device["text_input_ids"], batch_device["text_attention_mask"])
                temp_cls = nn.Linear(768, 3).to(DEVICE)
                logits = temp_cls(feat)
            else:
                feat = model(batch_device["image_pixel_values"])
                temp_cls = nn.Linear(768, 3).to(DEVICE)
                logits = temp_cls(feat)

            loss = criterion(logits, batch_device["label"])
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            true_labels = batch_device["label"].cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(true_labels)

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

    if save_bad_case and len(bad_cases) > 0:
        bad_case_df = pd.DataFrame(bad_cases)
        bad_case_path = os.path.join(OUTPUT_DIR, f"{modal_type}_bad_cases_early_fusion.csv")
        bad_case_df.to_csv(bad_case_path, index=False, encoding="utf-8")
        print(f"\n{modal_type.capitalize()} bad cases saved to: {bad_case_path}")
        print(f"Total {modal_type} bad cases: {len(bad_cases)}")

    val_loss = total_loss / len(loader)
    val_metrics = calculate_metrics(all_labels, all_preds)
    return val_loss, val_metrics


def train_single_modal(modal_type):
    print(f"\n=== Start Training {modal_type.capitalize()} Only Model (Ablation Study) ===")
    train_loader, val_loader, _ = get_data_loaders()
    model = TextBranch().to(DEVICE) if modal_type == "text" else ImageBranch().to(DEVICE)
    temp_classifier = nn.Linear(768, 3).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    # 合并参数
    if modal_type == "text":
        param_groups = [
            {"params": model.roberta.parameters(), "lr": BASE_LR * PRETRAIN_LR_RATIO, "weight_decay": WEIGHT_DECAY},
            {"params": model.feature_proj.parameters(), "lr": BASE_LR, "weight_decay": WEIGHT_DECAY},
            {"params": temp_classifier.parameters(), "lr": BASE_LR, "weight_decay": WEIGHT_DECAY}
        ]
    else:
        param_groups = [
            {"params": model.vit.parameters(), "lr": BASE_LR * PRETRAIN_LR_RATIO, "weight_decay": WEIGHT_DECAY},
            {"params": model.visual_attention.parameters(), "lr": BASE_LR * 1.2, "weight_decay": WEIGHT_DECAY},
            {"params": model.feature_proj.parameters(), "lr": BASE_LR, "weight_decay": WEIGHT_DECAY},
            {"params": temp_classifier.parameters(), "lr": BASE_LR, "weight_decay": WEIGHT_DECAY}
        ]
    optimizer = optim.AdamW(param_groups)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)
    # 适配低版本PyTorch：移除device_type参数
    scaler = GradScaler()

    best_macro_f1 = 0.0
    patience_counter = 0
    train_log = []
    best_model_path = os.path.join(OUTPUT_DIR, f"best_{modal_type}_model_early_fusion.pth")

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        # Training phase
        model.train()
        temp_classifier.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Train {modal_type.capitalize()}"):
            batch_device = {k: v.to(DEVICE) for k, v in batch.items() if k != "guid"}
            optimizer.zero_grad()
            with autocast():
                if modal_type == "text":
                    feat = model(batch_device["text_input_ids"], batch_device["text_attention_mask"])
                else:
                    feat = model(batch_device["image_pixel_values"])
                logits = temp_classifier(feat)
                loss = criterion(logits, batch_device["label"])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)
        scheduler.step()

        # Evaluation phase
        val_loss, val_metrics = eval_single_modal(
            model, val_loader, criterion, modal_type,
            save_bad_case=(epoch == EPOCHS - 1 or patience_counter >= PATIENCE - 1)
        )

        # Logging
        train_log.append({
            "epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, **val_metrics
        })

        # Print results
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Acc: {val_metrics['accuracy']:.4f} | Val Macro-F1: {val_metrics['macro_f1']:.4f}")

        # Early stopping
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "classifier_state_dict": temp_classifier.state_dict(),
                "best_metrics": val_metrics,
                "train_log": train_log,
                "epoch": epoch + 1
            }, best_model_path)
            patience_counter = 0
            print(f"Update Best {modal_type.capitalize()} Model! F1: {best_macro_f1:.4f} (Epoch {epoch + 1})")
        else:
            patience_counter += 1
            print(f"F1 No Improvement! Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                eval_single_modal(model, val_loader, criterion, modal_type, save_bad_case=True)
                print(f"Early Stopping Triggered! Best {modal_type.capitalize()} F1: {best_macro_f1:.4f}")
                break

    # 适配低版本PyTorch：移除weights_only参数（避免旧版本报错）
    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Load Best {modal_type.capitalize()} Model (Epoch {checkpoint['epoch']}, F1 {best_macro_f1:.4f})")

    # Save log
    pd.DataFrame(train_log).to_csv(os.path.join(OUTPUT_DIR, f"{modal_type}_train_log_early_fusion.csv"), index=False)
    print(f"{modal_type.capitalize()} Only Model Training Completed! Best Macro-F1: {best_macro_f1:.4f}")
    return best_macro_f1, train_log, model


def train_early_fusion():
    train_loader, val_loader, _ = get_data_loaders()
    model = EarlyFusionModel().to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    # Layer-wise learning rate
    param_groups = [
        {"params": model.text_branch.roberta.parameters(), "lr": BASE_LR * PRETRAIN_LR_RATIO,
         "weight_decay": WEIGHT_DECAY},
        {"params": model.image_branch.vit.parameters(), "lr": BASE_LR * PRETRAIN_LR_RATIO * 0.8,
         "weight_decay": WEIGHT_DECAY * 0.5},
        {"params": model.modal_align.parameters(), "lr": BASE_LR * 0.8, "weight_decay": WEIGHT_DECAY},
        {"params": model.image_gate.parameters(), "lr": BASE_LR * 0.8, "weight_decay": WEIGHT_DECAY},
        {"params": list(model.fusion_linear1.parameters()) + list(model.fusion_linear2.parameters()),
         "lr": BASE_LR, "weight_decay": WEIGHT_DECAY}
    ]
    optimizer = optim.AdamW(param_groups)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)
    # 适配低版本PyTorch：移除device_type参数
    scaler = GradScaler()

    best_macro_f1 = 0.0
    patience_counter = 0
    train_log = []
    attn_log = []
    best_model_path = os.path.join(OUTPUT_DIR, "best_early_fusion_model.pth")

    for epoch in range(EPOCHS):
        print(f"\n=== Epoch {epoch + 1}/{EPOCHS} (Early Fusion Optimized) ===")
        # Training
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler)
        # Evaluation
        save_bad_case = (epoch == EPOCHS - 1 or patience_counter >= PATIENCE - 1)
        val_loss, val_metrics, avg_attn = eval_epoch(
            model, val_loader, criterion,
            save_bad_case=save_bad_case,
            case_type="early_fusion"
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
        print(f"Average Attention Weights (Text/Image): {avg_attn[0]:.4f} / {avg_attn[1]:.4f}")
        print(f"Current Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        # Early stopping
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
            print(f"Update Best Early Fusion Model! F1: {best_macro_f1:.4f} (Epoch {epoch + 1})")
        else:
            patience_counter += 1
            print(f"F1 No Improvement! Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                eval_epoch(model, val_loader, criterion, save_bad_case=True, case_type="early_fusion")
                print(f"Early Stopping Triggered! Best Early Fusion F1: {best_macro_f1:.4f}")
                break

    # 适配低版本PyTorch：移除weights_only参数
    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Load Best Early Fusion Model (Epoch {checkpoint['epoch']}, F1 {best_macro_f1:.4f})")

    # Save logs
    pd.DataFrame(train_log).to_csv(os.path.join(OUTPUT_DIR, "early_fusion_train_log.csv"), index=False)
    pd.DataFrame(attn_log).to_csv(os.path.join(OUTPUT_DIR, "attention_log_early_fusion.csv"), index=False)
    print(f"\nEarly Fusion Model Training Completed! Best Val Macro-F1: {best_macro_f1:.4f}")
    return best_macro_f1, train_log, attn_log, model


# Test Set Prediction
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
    output_df.to_csv(os.path.join(OUTPUT_DIR, "test_result_early_fusion.txt"), index=False, header=False)
    print(f"\nTest Set Prediction Completed! Results saved to: {os.path.join(OUTPUT_DIR, 'test_result_early_fusion.txt')}")


# Visualization
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
    ax1.set_title("Early Fusion (Optimized) Training/Validation Loss Curves", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Subplot 2: Macro-F1 Curve
    ax2.plot(epochs, fusion_macro_f1, label="Early Fusion (Optimized) Macro-F1", color="green", marker="^", linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Macro-F1 Value", fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.set_title("Early Fusion (Optimized) Validation Macro-F1 Curve", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    # Subplot 3: Dynamic Attention Weights
    ax3.plot(attn_epochs, text_weight, label="Text Weight", color="purple", marker="o", linewidth=2)
    ax3.plot(attn_epochs, image_weight, label="Image Weight", color="orange", marker="s", linewidth=2)
    ax3.set_xlabel("Epoch", fontsize=12)
    ax3.set_ylabel("Attention Weight (0-1)", fontsize=12)
    ax3.set_title("Text/Image Dynamic Weight (Early Fusion Optimized)", fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)

    # Subplot 4: Ablation Study
    ax4.plot(epochs, fusion_macro_f1, label="Early Fusion (Optimized)", color="green", marker="^", linewidth=2)
    ax4.plot(text_epochs, text_macro_f1, label="Text Only", color="purple", marker="o", linewidth=2)
    ax4.plot(image_epochs, image_macro_f1, label="Image Only (With Visual Attention)", color="orange", marker="s",
             linewidth=2)
    ax4.set_xlabel("Epoch", fontsize=12)
    ax4.set_ylabel("Macro-F1 Value", fontsize=12)
    ax4.set_ylim(0, 1)
    ax4.set_title("Ablation Study: Early Fusion (Optimized) Macro-F1 Comparison", fontsize=14)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_curves_early_fusion.png"), dpi=300, bbox_inches="tight")
    print(f"\nTraining curves saved to: {os.path.join(OUTPUT_DIR, 'training_curves_early_fusion.png')}")


# Main Function
if __name__ == "__main__":
    # 训练优化版Early Fusion模型
    fusion_best_f1, fusion_log, attn_log, best_multi_model = train_early_fusion()
    # 训练单模态模型
    text_best_f1, text_log, _ = train_single_modal("text")
    image_best_f1, image_log, _ = train_single_modal("image")
    # 预测测试集
    predict_test_set(best_multi_model)
    # 可视化
    plot_training_curves(fusion_log, text_log, image_log, attn_log)

    print("\n===== Final Results Summary (Early Fusion Optimized Model) =====")
    print(f"Early Fusion (Optimized) Model Best Macro-F1: {fusion_best_f1:.4f}")
    print(f"Text Only Model Best Macro-F1: {text_best_f1:.4f}")
    print(f"Image Only Model Best Macro-F1: {image_best_f1:.4f}")
    print(f"\nAll output files are saved to: {OUTPUT_DIR}")
    print(f"\nBad case files (Early Fusion Optimized Model) saved in {OUTPUT_DIR}:")
    print("- early_fusion_bad_cases_early_fusion.csv (早期融合模型错误样本，含动态权重)")
    print("- text_bad_cases_early_fusion.csv (文本单模态错误样本)")
    print("- image_bad_cases_early_fusion.csv (图像单模态错误样本)")
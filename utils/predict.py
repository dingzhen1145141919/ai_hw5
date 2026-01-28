import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipModel, logging

# 添加项目根目录到Python路径，避免导入报错
sys.path.append("/media/ai/zcyStor/SATA/TY/hw5/models/blip_fusion")
# 关闭transformers警告输出
logging.set_verbosity_error()

# 路径配置 - 根据实际环境调整
ROOT_DIR = "/media/ai/zcyStor/SATA/TY/hw5"
DATA_DIR = os.path.join(ROOT_DIR, "data")
TEST_PATH = os.path.join(ROOT_DIR, "test_without_label.txt")
BLIP_PATH = os.path.join(ROOT_DIR, "models/blip-itm-base-coco")
MODEL_WEIGHT_PATH = "/media/ai/zcyStor/SATA/TY/hw5/models/blip_fusion/output_blip_3/best_blip_early_fusion_model.pth"
OUTPUT_PATH = os.path.join(ROOT_DIR, "test_with_pred_label.txt")  # 预测结果输出路径

# 模型/训练配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
TEXT_MAX_LEN = 64
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

# 模型定义 - 与训练时完全一致
class BLIPEarlyFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载BLIP预训练模型
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

        # 图像门控机制
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

    def forward(self, pixel_values, input_ids, attention_mask):
        # 获取BLIP基础特征
        outputs = self.blip(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # 兼容不同版本BLIP的特征提取
        try:
            img_feat = outputs.vision_model_output.pooler_output
            text_feat = outputs.text_model_output.pooler_output
        except:
            img_feat = outputs.vision_model_output.last_hidden_state[:, 0, :]
            text_feat = outputs.text_model_output.last_hidden_state[:, 0, :]

        # 特征归一化
        text_feat = torch.nn.functional.normalize(text_feat, p=2, dim=1)
        img_feat = torch.nn.functional.normalize(img_feat, p=2, dim=1)

        # 模态对齐
        img_feat = self.modal_align(img_feat)

        # 图像门控
        gate_weight = self.image_gate(text_feat)
        img_feat = img_feat * gate_weight

        # 特征融合
        fused_feat = torch.cat([text_feat, img_feat], dim=1)
        x = self.fusion_linear1(fused_feat)
        x = self.fusion_norm(x)
        x = self.relu(x)
        x = self.fusion_dropout(x)
        x = x + text_feat  # 残差连接
        logits = self.fusion_linear2(x)

        return logits

# 测试集数据集定义
class TestDataset(Dataset):
    def __init__(self, test_df, processor):
        self.test_df = test_df
        self.processor = processor

    def __len__(self):
        return len(self.test_df)

    def __getitem__(self, idx):
        row = self.test_df.iloc[idx]
        guid = str(row["guid"])

        # 加载图像，失败则用空白图兜底
        try:
            img = Image.open(os.path.join(DATA_DIR, f"{guid}.jpg")).convert("RGB")
        except Exception as e:
            print(f"警告：图像 {guid}.jpg 读取失败，使用空白图 | 错误：{e}")
            img = Image.new('RGB', (224, 224), color='white')

        # 加载文本，优先pro文本，无则用基础txt
        text = ""
        # 尝试读取pro文本
        pro_txt_path = os.path.join(DATA_DIR, f"{guid}_pro.txt")
        if os.path.exists(pro_txt_path):
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
            for encoding in encodings:
                try:
                    with open(pro_txt_path, 'r', encoding=encoding, errors="ignore") as f:
                        text = f.read().strip()
                    break
                except:
                    continue
        # 若无pro文本，读取基础txt
        if not text:
            txt_path = os.path.join(DATA_DIR, f"{guid}.txt")
            if os.path.exists(txt_path):
                encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
                for encoding in encodings:
                    try:
                        with open(txt_path, 'r', encoding=encoding, errors="ignore") as f:
                            text = f.read().strip()
                        break
                    except:
                        continue
        # 空文本填充默认值
        if not text:
            text = "empty"

        # BLIP编码
        encoding = self.processor(
            images=img,
            text=text,
            padding="max_length",
            truncation=True,
            max_length=TEXT_MAX_LEN,
            return_tensors="pt"
        )

        return {
            "guid": guid,
            "pixel_values": encoding["pixel_values"].squeeze(),
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze()
        }

# 核心预测函数
def predict_null_labels():
    # 功能：加载模型预测test集null标签，替换后保存
    # 加载BLIP处理器
    print("加载BLIP处理器...")
    processor = BlipProcessor.from_pretrained(
        BLIP_PATH,
        local_files_only=True,
        ignore_mismatched_sizes=True
    )

    # 读取测试集，兼容多编码格式
    print("读取测试集文件...")
    def read_csv_compatible(file_path):
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
        for enc in encodings:
            try:
                return pd.read_csv(file_path, encoding=enc, on_bad_lines='skip')
            except:
                continue
        return pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')

    test_df = read_csv_compatible(TEST_PATH)
    # 确保guid为字符串类型
    test_df["guid"] = test_df["guid"].astype(str)
    # 填充空标签为null
    test_df["tag"] = test_df["tag"].fillna("null")
    print(f"测试集样本总数：{len(test_df)}")
    print(f"含null标签的样本数：{len(test_df[test_df['tag'] == 'null'])}")

    # 构建测试集加载器
    test_dataset = TestDataset(test_df, processor)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # 初始化模型并加载权重
    print("加载最优模型权重...")
    model = BLIPEarlyFusionModel().to(DEVICE)
    # 加载checkpoint
    checkpoint = torch.load(MODEL_WEIGHT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # 切换到评估模式
    print(f"模型权重加载完成！最优模型F1：{checkpoint['best_metrics']['macro_f1']:.4f}")

    # 批量预测
    print("开始预测测试集标签...")
    all_guids = []
    all_pred_tags = []
    with torch.no_grad():  # 禁用梯度计算，节省显存
        for batch in tqdm(test_loader, desc="Predicting"):
            guids = batch["guid"]
            # 数据移到指定设备
            batch_device = {
                "pixel_values": batch["pixel_values"].to(DEVICE),
                "input_ids": batch["input_ids"].to(DEVICE),
                "attention_mask": batch["attention_mask"].to(DEVICE)
            }
            # 模型预测
            logits = model(
                pixel_values=batch_device["pixel_values"],
                input_ids=batch_device["input_ids"],
                attention_mask=batch_device["attention_mask"]
            )
            # 取预测概率最大的标签
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            pred_tags = [ID2LABEL[pred] for pred in preds]

            all_guids.extend(guids)
            all_pred_tags.extend(pred_tags)

    # 替换null标签并保存
    print("替换null标签并保存结果...")
    # 构建预测结果DataFrame
    pred_df = pd.DataFrame({
        "guid": all_guids,
        "pred_tag": all_pred_tags
    })
    # 合并原始测试集和预测结果
    test_df["guid"] = test_df["guid"].astype(str)
    pred_df["guid"] = pred_df["guid"].astype(str)
    merged_df = test_df.merge(pred_df, on="guid", how="left")

    # 替换null标签：原始为null的用预测值，非null保留原值
    merged_df["final_tag"] = merged_df.apply(
        lambda row: row["pred_tag"] if row["tag"] == "null" else row["tag"],
        axis=1
    )

    # 保留原始列结构，仅替换tag列
    final_df = merged_df[["guid"]].copy()
    final_df["tag"] = merged_df["final_tag"]

    # 保存最终结果
    final_df.to_csv(OUTPUT_PATH, index=False, header=True, encoding="utf-8")
    print(f"预测完成！结果保存至：{OUTPUT_PATH}")

    # 打印统计信息
    print("\n=== 预测结果统计 ===")
    print(f"总样本数：{len(final_df)}")
    print(f"替换null标签数：{len(test_df[test_df['tag'] == 'null'])}")
    print(f"预测标签分布：")
    print(final_df["tag"].value_counts())

    return final_df

# 主函数
if __name__ == "__main__":
    # 显存优化
    torch.cuda.empty_cache()
    try:
        # 执行预测
        result_df = predict_null_labels()
        print("\n所有null标签替换完成！")
    except Exception as e:
        print(f"\n预测过程出错：{str(e)}")
        import traceback
        traceback.print_exc()
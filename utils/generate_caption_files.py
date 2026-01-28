import os
import re
import torch
import pandas as pd
import random
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration, logging

# 关闭警告输出
logging.set_verbosity_error()
import warnings
warnings.filterwarnings("ignore")

# 路径配置 - 需根据实际环境调整
CAPTION_MODEL_PATH = "/media/ai/zcyStor/SATA/TY/hw5/models/blip-image-captioning-base"
ROOT_DIR = "/media/ai/zcyStor/SATA/TY/hw5"
TRAIN_LABEL_PATH = os.path.join(ROOT_DIR, "train.txt")
DATA_DIR = os.path.join(ROOT_DIR, "data")

# 处理配置
SKIP_EXISTING = False  # 跳过已生成的_pro.txt文件
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8  # 按显存调整（8G=8，16G=16，4G=4）
SEED = 42
IMAGE_SIZE = 224

# 固定随机种子保证结果可复现
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
set_seed(SEED)

# 兼容多编码读取文件
def safe_read_csv(file_path):
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except Exception as e:
            continue
    return pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')

# 清理生成的描述文本
def clean_caption(text):
    if not isinstance(text, str) or text.strip() == "":
        return "a clear image with various contents"
    # 保留英文有效字符
    text = re.sub(r"[^a-zA-Z0-9\s.,!?\-:()]", "", text)
    # 合并多余空格
    text = re.sub(r"\s+", " ", text).strip()
    # 补充默认描述
    return text if len(text) > 5 else "a clear image with various contents"

# 获取train.txt中的有效guid列表
def get_all_train_guids():
    train_df = safe_read_csv(TRAIN_LABEL_PATH)
    train_df["guid"] = train_df["guid"].astype(str).str.strip()
    valid_guids = train_df[train_df["guid"] != "nan"]["guid"].unique().tolist()
    print(f"有效guid数量：{len(valid_guids)}")
    return valid_guids

# 生成xxx_pro.txt文件（原有内容+图片描述）
def full_generate_pro_txt():
    # 加载BLIP模型
    print(f"\n加载模型：{CAPTION_MODEL_PATH}")
    try:
        processor = BlipProcessor.from_pretrained(CAPTION_MODEL_PATH, local_files_only=True)
        model = BlipForConditionalGeneration.from_pretrained(
            CAPTION_MODEL_PATH,
            local_files_only=True,
            ignore_mismatched_sizes=True
        ).to(DEVICE)
        model.eval()
        print("模型加载完成")
    except Exception as e:
        print(f"模型加载失败：{e}")
        print("检查模型路径及核心文件（pytorch_model.bin/config.json）")
        return

    # 获取待处理guid
    all_guids = get_all_train_guids()
    if not all_guids:
        print("无有效guid，终止处理")
        return

    # 过滤已生成文件
    if SKIP_EXISTING:
        pending_guids = []
        for guid in all_guids:
            pro_txt_path = os.path.join(DATA_DIR, f"{guid}_pro.txt")
            if not os.path.exists(pro_txt_path):
                pending_guids.append(guid)
        print(f"待处理guid数量：{len(pending_guids)}")
        all_guids = pending_guids
        if not all_guids:
            print("所有文件已生成，无需处理")
            return

    # 批次处理
    processed_count = 0
    failed_count = 0
    skipped_count = 0

    for idx in tqdm(range(0, len(all_guids), BATCH_SIZE), desc="生成描述文件"):
        batch_guids = all_guids[idx:idx+BATCH_SIZE]
        batch_images = []
        batch_original_texts = []
        valid_guids = []

        # 加载批次图片和文本
        for guid in batch_guids:
            img_path = os.path.join(DATA_DIR, f"{guid}.jpg")
            if not os.path.exists(img_path):
                skipped_count += 1
                print(f"\n跳过：图片不存在 {img_path}")
                continue

            # 读取原始文本
            original_txt_path = os.path.join(DATA_DIR, f"{guid}.txt")
            try:
                with open(original_txt_path, "r", encoding="utf-8", errors="ignore") as f:
                    original_text = f.read().strip()
            except:
                original_text = ""

            # 加载图片
            try:
                image = Image.open(img_path).convert("RGB")
                batch_images.append(image)
                batch_original_texts.append(original_text)
                valid_guids.append(guid)
            except Exception as e:
                failed_count += 1
                print(f"\n失败：图片加载错误 {img_path}，{e}")
                continue

        if not valid_guids:
            continue

        # 批量生成描述
        inputs = processor(
            images=batch_images,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                pixel_values=inputs.pixel_values,
                max_length=32,
                num_beams=4,
                repetition_penalty=1.2,
                do_sample=False,
                early_stopping=True
            )

        # 解码并写入文件
        raw_captions = processor.batch_decode(outputs, skip_special_tokens=True)
        clean_captions = [clean_caption(cap) for cap in raw_captions]

        for guid, original_text, caption in zip(valid_guids, batch_original_texts, clean_captions):
            final_text = f"{original_text} {caption}".strip() if original_text else caption
            pro_txt_path = os.path.join(DATA_DIR, f"{guid}_pro.txt")
            with open(pro_txt_path, "w", encoding="utf-8") as f:
                f.write(final_text)
            processed_count += 1

    # 输出处理统计
    print(f"\n处理完成！")
    print(f"成功生成：{processed_count} 个文件")
    print(f"跳过（无图片）：{skipped_count} 个")
    print(f"失败（加载错误）：{failed_count} 个")
    print(f"总计：{processed_count + skipped_count + failed_count} 个guid")

    # 验证示例
    if processed_count > 0:
        sample_guid = all_guids[0]
        sample_pro_txt = os.path.join(DATA_DIR, f"{sample_guid}_pro.txt")
        if os.path.exists(sample_pro_txt):
            with open(sample_pro_txt, "r", encoding="utf-8") as f:
                sample_content = f.read().strip()
            print(f"\n示例（{sample_guid}）：")
            print(f"内容：{sample_content}")

# 主函数
if __name__ == "__main__":
    # 前置检查
    if not os.path.exists(TRAIN_LABEL_PATH):
        print(f"train.txt不存在：{TRAIN_LABEL_PATH}")
    elif not os.path.exists(DATA_DIR):
        print(f"数据目录不存在：{DATA_DIR}")
    else:
        torch.cuda.empty_cache()
        full_generate_pro_txt()
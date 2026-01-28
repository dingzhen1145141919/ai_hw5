import os
import sys
import pandas as pd
# 导入语言检测函数
from count_language import is_english_text

def filter_train_set():
    # 筛选训练集：保留英语文本且图文文件存在的样本
    # 路径配置（适配utils/ → hw5/目录结构）
    TRAIN_PATH = "../train.txt"  # 原始训练集
    DATA_DIR = "../data"         # 图文文件目录
    CLEANED_TRAIN_PATH = "../train_english_only.txt"  # 筛选后训练集

    # 检查文件路径
    if not os.path.exists(TRAIN_PATH):
        print(f"错误：训练集文件不存在 {TRAIN_PATH}")
        sys.exit(1)
    if not os.path.exists(DATA_DIR):
        print(f"错误：data文件夹不存在 {DATA_DIR}")
        sys.exit(1)

    # 读取原始数据
    train_df = pd.read_csv(TRAIN_PATH)
    print(f"原始样本数：{len(train_df)}")

    # 筛选有效样本
    valid_guids = []
    for _, row in train_df.iterrows():
        guid = str(row["guid"])
        # 检查对应的txt和jpg文件
        txt_path = os.path.join(DATA_DIR, f"{guid}.txt")
        img_path = os.path.join(DATA_DIR, f"{guid}.jpg")
        if not (os.path.exists(txt_path) and os.path.exists(img_path)):
            continue
        # 检测文本是否为英语
        if is_english_text(txt_path):
            valid_guids.append(guid)

    # 生成筛选后的训练集
    train_cleaned = train_df[train_df["guid"].astype(str).isin(valid_guids)]
    print(f"筛选后英语样本数：{len(train_cleaned)}")
    print(f"剔除样本数：{len(train_df) - len(train_cleaned)}")

    # 保存文件
    train_cleaned.to_csv(CLEANED_TRAIN_PATH, index=False, header=True)
    print(f"\n筛选完成！文件保存至：{CLEANED_TRAIN_PATH}")

if __name__ == "__main__":
    filter_train_set()
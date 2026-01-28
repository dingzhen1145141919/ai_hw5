import os
import sys
from langdetect import detect, LangDetectException

def is_english_text(txt_path):
    # 判断txt文件内容是否为英语（供其他脚本调用）
    try:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()[:200]  # 取前200字符平衡效率和准确性
        if not text:
            return False  # 空文本判定为非英语
        return detect(text) == "en"
    except LangDetectException:
        return False  # 无法检测语言判定为非英语
    except Exception:
        return False  # 其他异常判定为非英语

def count_text_languages():
    # 统计data目录下txt文件的语言分布
    DATA_DIR = "../data"
    if not os.path.exists(DATA_DIR):
        print(f"错误：data文件夹不存在 {DATA_DIR}")
        print("请确认脚本在 hw5/utils/ 目录下运行")
        sys.exit(1)

    # 获取所有txt文件
    txt_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")]
    if not txt_files:
        print("错误：data文件夹下无txt文件")
        sys.exit(1)

    # 统计各语言数量
    lang_stat = {}
    undetectable = 0
    empty = 0
    print(f"开始统计 {len(txt_files)} 个txt文件的语言分布...")
    for filename in txt_files:
        txt_path = os.path.join(DATA_DIR, filename)
        try:
            with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()[:200]
            if not text:
                lang = "empty"
                empty += 1
            else:
                lang = detect(text)
            lang_stat[lang] = lang_stat.get(lang, 0) + 1
        except LangDetectException:
            undetectable += 1
        except Exception:
            undetectable += 1

    # 输出统计结果
    print("\n" + "="*50)
    print("文本语言分布统计")
    print("="*50)
    sorted_lang = sorted(lang_stat.items(), key=lambda x: x[1], reverse=True)
    for lang, count in sorted_lang:
        lang_desc = {"en": "英语", "empty": "空文本"}.get(lang, "其他语言")
        ratio = (count / len(txt_files)) * 100
        print(f"{lang:5s}（{lang_desc}）: {count:4d} 个（{ratio:.2f}%）")
    print(f"无法检测语言：{undetectable:4d} 个（{undetectable/len(txt_files)*100:.2f}%）")
    print("\n" + "="*50)
    print(f"核心结论：英语样本占比 {lang_stat.get('en', 0)/len(txt_files)*100:.2f}%")

if __name__ == "__main__":
    count_text_languages()
import json
import os
import pandas as pd
from collections import defaultdict

def process_emoset_to_csv(dataset_root_path: str, output_csv_path: str):
    """
    遍历 EmoSet 的 annotation 文件夹，提取亮度和色彩度，
    并将其汇总计算为 pandas DataFrame 支持的 CSV 格式。
    """
    annotation_folder = os.path.join(dataset_root_path, "annotation")
    
    if not os.path.exists(annotation_folder):
        print(f"❌ 错误: 找不到文件夹 {annotation_folder}")
        print("请确保你提供的路径下包含 'annotation' 这个文件夹。")
        return
        
    print(f"🔍 开始扫描 EmoSet 数据集: {annotation_folder}")
    print("⏳ 数据量较大 (约 11.8 万个文件)，请耐心等待...")

    # 用于累加数值和计数的字典
    emotion_stats = defaultdict(lambda: {"brightness_sum": 0.0, "colorfulness_sum": 0.0, "count": 0})
    processed_count = 0
    error_count = 0

    # 递归遍历所有子文件夹和文件
    for root, _, files in os.walk(annotation_folder):
        for filename in files:
            if filename.endswith(".json"):
                filepath = os.path.join(root, filename)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        emotion = data.get("emotion")
                        brightness = data.get("brightness")
                        colorfulness = data.get("colorfulness")
                        
                        # 确保关键数据存在
                        if emotion and brightness is not None and colorfulness is not None:
                            # 统一转为小写，防止大小写带来的重复分类
                            emotion = emotion.strip().lower()
                            emotion_stats[emotion]["brightness_sum"] += float(brightness)
                            emotion_stats[emotion]["colorfulness_sum"] += float(colorfulness)
                            emotion_stats[emotion]["count"] += 1
                            
                    processed_count += 1
                    
                    # 打印进度指示器
                    if processed_count % 10000 == 0:
                        print(f"   已处理 {processed_count} 个 JSON 标注文件...")
                        
                except Exception as e:
                    error_count += 1
                    pass # 忽略格式损坏的文件
                    
    print(f"\n✅ 扫描完成！共成功处理 {processed_count} 个文件 (失败 {error_count} 个)。")
    print("⚙️ 正在计算均值并生成 CSV...")

    # 整合结果，计算均值
    records = []
    for emo, stats in emotion_stats.items():
        if stats["count"] > 0:
            avg_brightness = stats["brightness_sum"] / stats["count"]
            avg_colorfulness = stats["colorfulness_sum"] / stats["count"]
            
            records.append({
                "emotion": emo,
                "brightness_mean": round(avg_brightness, 4), # 保留 4 位小数保证精度
                "colorfulness_mean": round(avg_colorfulness, 4)
            })

    # 将结果转换为 Pandas DataFrame 并保存为 CSV
    df = pd.DataFrame(records)
    
    # 按情感名称排序，保持输出整洁
    df.sort_values(by="emotion", inplace=True)
    
    # 确保保存的目录存在
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    df.to_csv(output_csv_path, index=False, encoding='utf-8')
    print(f"🎉 成功！特征汇总文件已保存至: {output_csv_path}")
    
    # 打印前几行预览
    print("\n📊 提取的数据预览:")
    print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    # ==========================================
    # 🔴 注意：在运行前，请修改这里的 DATASET_ROOT 路径
    # 指向你解压出来的 EmoSet 文件夹（里面应该包含 annotation 文件夹）
    # ==========================================
    DATASET_ROOT = "./EmoSet" 
    
    # 我们的 Agent 代码中期待读取的输出路径
    OUTPUT_CSV = "data/processed/emoset_emotion_summary.csv"
    
    process_emoset_to_csv(DATASET_ROOT, OUTPUT_CSV)

import json
import argparse
from pathlib import Path
from tqdm import tqdm

def filter_media_tokens(input_file, output_file):
    """
    过滤包含<image>、<video>或<audio>标记的数据
    
    Args:
        input_file: 输入的jsonl或json文件路径
        output_file: 过滤后的输出文件路径
    """
    # 确定文件格式
    is_jsonl = input_file.suffix == '.jsonl'
    
    print(f"正在处理文件: {input_file}")
    
    # 读取数据
    data = []
    if is_jsonl:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    else:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    
    print(f"原始数据记录数: {len(data)}")
    
    # 媒体标记列表
    media_tokens = ["<image>", "<video>", "<audio>"]
    
    # 过滤数据
    filtered_data = []
    for item in tqdm(data, desc="过滤数据"):
        # 获取所有需要检查的字段
        input_field = item.get("input", "")
        instruction_field = item.get("instruction", "")
        output_field = item.get("output", item.get("response", item.get("answer", "")))
        
        # 新的过滤条件：存在history字段 或 任意字段包含媒体标记
        if "history" in item or any(
            token in str(field)
            for field in [input_field, instruction_field, output_field]
            for token in media_tokens
        ):
            continue
        
        filtered_data.append(item)
    
    # 保存过滤后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in filtered_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    filtered_count = len(data) - len(filtered_data)
    print(f"过滤掉的记录数: {filtered_count}")
    print(f"保留的记录数: {len(filtered_data)}")
    print(f"过滤后的数据已保存至: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="过滤包含媒体标记的数据集")
    parser.add_argument("--input", "-i", type=str, required=True, help="输入文件路径 (.json 或 .jsonl)")
    parser.add_argument("--output", "-o", type=str, required=True, help="输出文件路径")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # 检查输入文件是否存在
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    
    # 创建输出目录（如果不存在）
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    filter_media_tokens(input_path, output_path)

if __name__ == "__main__":
    main() 
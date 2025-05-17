import os
import typer
from pathlib import Path
from transformers import AutoTokenizer
from omnisvg_data_process.tokenizer import SVGTokenizer  # 假设SVGTokenizer可访问

# 创建Typer应用实例
app = typer.Typer()

def load_base_tokenizer(model_path: str):
    """加载基础的Hugging Face tokenizer"""
    print(f"从{model_path}加载基础tokenizer")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        print(f"原始词汇表大小: {tokenizer.vocab_size}")
        return tokenizer
    except Exception as e:
        print(f"加载tokenizer时出错: {e}")
        raise typer.Exit(code=1)

def get_svg_tokens():
    """初始化SVGTokenizer并返回其词汇表tokens"""
    print("初始化SVGTokenizer...")
    try:
        svg_tokenizer_instance = SVGTokenizer()
        svg_tokens = sorted(list(svg_tokenizer_instance.vocabulary))
        print(f"SVG词汇表大小: {len(svg_tokens)}")
        return svg_tokens
    except Exception as e:
        print(f"初始化SVGTokenizer或构建词汇表时出错: {e}")
        raise typer.Exit(code=1)

def add_tokens_to_tokenizer(tokenizer, new_tokens):
    """向tokenizer添加新tokens"""
    print("添加tokens...")
    
    # 获取添加前的词汇表大小
    original_size = len(tokenizer)
    
    # 添加新tokens
    num_added = tokenizer.add_tokens(new_tokens, special_tokens=False)
    
    # 获取添加后的词汇表大小
    new_size = len(tokenizer)
    
    print(f"添加了{num_added}个新tokens")
    print(f"添加前词汇表大小: {original_size}")
    print(f"添加后词汇表大小: {new_size}")
    
    return tokenizer

def verify_tokenizer(tokenizer, test_tokens):
    """验证tokenizer中的tokens是否正确添加"""
    print("验证更新后的tokenizer...")
    
    for token in test_tokens[:5]:  # 只测试前5个token
        token_id = tokenizer.convert_tokens_to_ids(token)
        reconstructed = tokenizer.convert_ids_to_tokens(token_id)
        
        if token != reconstructed:
            print(f"警告: token不匹配! '{token}' -> '{reconstructed}'")
        else:
            print(f"验证成功: '{token}' -> ID {token_id} -> '{reconstructed}'")

@app.command()
def add_tokens(
    base_model_path: Path = typer.Argument(..., help="基础Hugging Face tokenizer目录路径"),
    output_dir: Path = typer.Argument(..., help="保存更新后tokenizer的目录")
):
    """向现有tokenizer添加SVG tokens并保存结果"""
    
    # 1. 加载基础tokenizer
    tokenizer = load_base_tokenizer(str(base_model_path))
    
    # 2. 获取SVG tokens
    svg_tokens = get_svg_tokens()
    
    # 3. 添加tokens到tokenizer
    updated_tokenizer = add_tokens_to_tokenizer(tokenizer, svg_tokens)
    
    # 4. 保存更新后的tokenizer
    print(f"将更新后的tokenizer保存到: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    updated_tokenizer.save_pretrained(str(output_dir))
    
    # 5. 验证
    verify_tokenizer(updated_tokenizer, svg_tokens)
    
    print("Token添加过程完成")

if __name__ == "__main__":
    app()

# OmniSVG: 统一的可扩展矢量图形生成模型

OmniSVG 是一个统一的框架，利用预训练的视觉语言模型 (VLM) 实现端到端的多模态可缩放矢量图形 (SVG) 生成。它通过将 SVG 命令和坐标参数化为离散的词元 (token)，将结构逻辑与底层几何解耦，从而实现高效训练，同时保持复杂 SVG 结构的表达能力。

## 主要特性

*   **高质量SVG生成**: 能够自回归地生成高质量SVG，涵盖从简单图标到复杂动漫角色的广泛复杂度。
*   **多种生成模式**: 支持文本到SVG (Text-to-SVG)、图像到SVG (Image-to-SVG) 和角色参考SVG (Character-Reference SVG) 生成。
*   **端到端多模态**: 首个利用预训练VLM的端到端多模态SVG生成家族。
*   **MMSVG-2M数据集**: 发布了包含两百万SVG资产的大规模数据集 MMSVG-2M 及标准化评测协议。

## MMSVG-2M 数据集

MMSVG-2M 是一个大规模 SVG 数据集，包含两百万个 SVG 样本，涵盖图标、插画和动漫设计。数据经过整理、简化（使用原子命令如 M, L, C, A, Z, F）和标注，用于训练和评估 SVG 生成模型。

## OmniSVG 模型

OmniSVG 基于预训练的视觉语言模型 (如 Qwen2.5-VL)，并集成了一个专门的 SVG 词元分析器 (tokenizer)。

### SVG 参数化与词元化

根据论文和相关讨论，OmniSVG 的 SVG 词元化关键点如下：

*   **原子命令**: SVG 被简化为一系列原子命令（Move To (M)、Line To (L)、Cubic Bézier (C)、Elliptical Arc (A)、ClosePath (Z)）和填充命令 (Fill (F))。
*   **坐标编码**: 坐标值 (例如，在 200x200 的画布上，0≤x, y≤199) 被离散化并通过特定映射（如 `position_token = canvas_width * x + y`）编码为单一的词元。这有助于模型学习几何关系，并避免了将坐标数字作为单独词元处理时可能出现的 "坐标幻觉" 问题。
*   **颜色编码**: RGB 颜色值被量化（例如，每个通道量化为4位，即16个级别），然后组合编码为单一的颜色词元。
*   **序列化**: 完整的SVG表示为一个包含命令词元、参数化坐标词元和参数化颜色词元的序列。特殊词元如 `<SOP>` (Start Of Path/Picture) 和 `<EOS>` (End Of SVG) 用于标记序列的开始和结束。
*   **模型输入**: 文本或图像等多模态输入首先被其各自的tokenizer处理成前缀词元 (prefix tokens)，然后SVG词元序列拼接到这些前缀词元之后，形成完整的输入序列送入VLM。

这种词元化方法使得模型能够高效地学习SVG的结构和视觉属性。

### 模型架构

OmniSVG 利用了像 Qwen2.5-VL 这样的预训练视觉语言模型。其核心思想是将SVG的生成视为一个序列到序列的任务。
1.  **输入处理**: 文本提示或图像输入被相应的编码器处理成嵌入向量。
2.  **SVG词元化**: 目标SVG图像被 `omnisvg_data_process.tokenizer.SVGTokenizer` 转换成上述的词元序列。
3.  **自回归生成**: 模型以自回归的方式预测SVG词元序列，条件是给定的文本/图像输入。

## 代码库结构

*   `omnisvg_data_process/`: 包含数据处理的核心逻辑。
    *   `tokenizer.py`: 核心的 `SVGTokenizer` 实现，负责将SVG原始数据转换为模型可以理解的词元序列，包括命令、参数化坐标和颜色。
    *   `normalizer.py`: 用于SVG内容的规范化，例如简化路径、转换命令等。
    *   `constants.py`: 定义项目中使用的常量，如SVG画布的默认视图框大小 (`VIEWBOX_WIDTH`, `VIEWBOX_HEIGHT`)。
    *   `entokenize.py` / `detokenize.py`: 可能包含将文本描述或SVG内容与词元双向转换的辅助脚本或逻辑。
    *   `utils.py`: 通用工具函数。
*   `scripts/`: 包含用于数据集构建、模型预处理/后处理和推理测试的脚本。

## 主要脚本功能

*   **`scripts/build_sft_dataset.py`**:
    *   用于构建监督微调 (SFT) 数据集。
    *   从多种格式 (CSV, JSON, JSONL) 的输入文件中加载SVG数据及其文本描述。
    *   对SVG内容进行规范化 (使用 `omnisvg_data_process.tokenizer.SVGTokenizer.normalize_svg`)。
    *   将规范化后的SVG词元化 (使用 `omnisvg_data_process.tokenizer.SVGTokenizer.svg_to_tokens`)。
    *   将SVG栅格化为PNG图像 (使用 `cairosvg`)，作为多模态输入的一部分或用于可视化。
    *   生成包含文本指令、图像（可选）和目标SVG词元序列的训练样本，保存为 JSON Lines 格式。

*   **`scripts/merge_tokenizer.py`**:
    *   将一个基础的BPE (Byte Pair Encoding) 词元分析器 (例如来自预训练的LLM，如Qwen) 与 `omnisvg_data_process.tokenizer.SVGTokenizer` 定义的SVG特定词元（包括参数化的坐标和颜色词元）进行合并。
    *   目的是扩展基础模型的词汇表，使其能够理解和生成SVG特定的词元。
    *   保存合并后的词元分析器文件，以便后续加载到模型中。

*   **`scripts/resize_model_embeddings.py`**:
    *   在词元分析器词汇表扩大后（由于合并了SVG词元），此脚本用于调整预训练模型的词嵌入层 (token embedding layer) 和可能的输出层 (LM head) 的大小，以匹配新的词汇表大小。
    *   加载基础模型 (如 `Qwen2_5_VLForConditionalGeneration`)。
    *   调用 `model.resize_token_embeddings()` 方法。
    *   如果需要，手动调整LM head的权重。
    *   保存调整大小后的模型。

*   **`scripts/inference_test.py`**:
    *   加载经过词元分析器合并和模型嵌入层调整后的模型 (如 `Qwen2_5_VLForConditionalGeneration`) 和相应的词元分析器、处理器。
    *   接收文本提示 (以及可选的图像输入)。
    *   对输入进行预处理，应用聊天模板。
    *   使用模型进行推理，生成SVG（或其他形式的输出，取决于模型的训练）。
    *   对生成的词元进行解码并打印输出。

## 如何引用

如果您的研究中使用了 OmniSVG 模型或 MMSVG-2M 数据集，请考虑引用原始论文：

```bibtex
@article{yang2024omnisvg,
  title={OmniSVG: A Unified Scalable Vector Graphics Generation Model},
  author={Yang, Yiying and Cheng, Wei and Chen, Sijin and Zeng, Xianfang and Zhang, Jiaxu and Wang, Liao and Yu, Gang and Ma, Xingjun and Jiang, Yu-Gang},
  journal={arXiv preprint arXiv:2404.06263},
  year={2024}
}
```
*(请注意：BibTeX条目可能会随论文正式发表而更新，请以官方渠道为准。)*

## 相关链接

*   **项目主页**: [http://omnisvg.github.io](http://omnisvg.github.io)
*   **ArXiv论文**: [https://arxiv.org/abs/2404.06263](https://arxiv.org/abs/2404.06263)

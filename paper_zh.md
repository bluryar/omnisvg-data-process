# OmniSVG：统一的可扩展矢量图形生成模型

Yiying Yang1,2∗ Wei Cheng2∗ Sijin Chen1 Xianfang Zeng2 Jiaxu Zhang2 Liao Wang2 Gang Yu2 ‡ Xingjun Ma1 ‡ Yu-Gang Jiang1 1 Fudan University 2 StepFun http://omnisvg.github.io

![](https://cdn-mineru.openxlab.org.cn/extract/bd462ee4-da41-4d04-8103-15f5a4059566/ec6a8472d293f438a001c04b50439dff42a8818f4040dce7e507c0945a6e91b3.jpg)
图1. OmniSVG的亮点特性。OmniSVG能够自回归地生成高质量SVG，涵盖从简单图标到复杂动漫角色的广泛复杂度。它通过多种生成模式展现出卓越的多样性，包括Text-to-SVG、Image-to-SVG和Character-Reference SVG，使其成为多样化创意任务的强大灵活解决方案。

# 摘要

可扩展矢量图形（SVG）是一种因其分辨率无关性和可编辑性而被广泛应用于图形设计的重要图像格式。高质量SVG的生成研究持续受到设计师和AIGC社区研究者的关注。然而，现有方法要么输出无结构结果且计算成本巨大，要么仅能生成过于简化结构的单色图标。为生成高质量且复杂的SVG，我们提出了OmniSVG，一个统一框架，利用预训练的Vision-Language Models（VLMs）实现端到端的多模态SVG生成。通过将SVG命令和坐标参数化为离散token，OmniSVG将结构逻辑与底层几何解耦，实现高效训练的同时保持复杂SVG结构的表达能力。为进一步推动SVG合成的发展，我们引入了MMSVG-2M，这是一个包含两百万丰富标注SVG资产的多模态数据集，并提出了标准化的条件SVG生成评测协议。大量实验表明，OmniSVG优于现有方法，并展现出集成到专业SVG设计工作流的潜力。

# 1. 引言

可扩展矢量图形（SVG）因其分辨率无关性、文件体积小和天然可编辑性，已成为现代数字设计的基石。SVG被广泛应用于从UI/UX设计到工业CAD系统的专业工作流，能够精确操作几何基元（如Bezier曲线、多边形），在不同分辨率下保持高精度和一致的视觉质量。然而，对于非专业人士来说，创建高质量SVG内容仍然具有挑战性，需要掌握专业工具或复杂的XML语法。

现有生成SVG内容的方法主要采用优化型方法或自回归方法。

优化型方法[19, 30, 56]通过优化可微分矢量图形光栅化器，迭代地细化SVG参数。虽然这些方法在生成SVG图标方面有效，但在处理复杂样本时计算开销巨大，且输出结构无序、锚点冗余。相比之下，自回归方法构建transformer模型或适配预训练大语言模型（LLMs），直接生成XML参数[55]或代码[37, 53]来表示SVG。得益于端到端学习流程，自回归方法更具可扩展性[5]，能够直接从大量SVG样本中学习。然而，现有自回归方法受限于上下文窗口长度和复杂SVG数据的稀缺性[9, 20, 49]，难以处理高复杂度SVG。为解决这些限制，我们提出了OmniSVG，这是首个利用预训练VLMs实现端到端多模态复杂SVG生成的统一框架。通过将SVG坐标和命令参数化为离散token，OmniSVG将结构逻辑与底层几何解耦，缓解了基于代码的LLMs常见的"坐标幻觉"问题，并能生成生动多彩的SVG结果。得益于下一个token预测训练目标，OmniSVG还能在给定部分观测的情况下完成SVG，生成多样化结果。与传统自回归SVG生成相比，我们的方法可将SVG参数化为最长达3万token，支持复杂高质量SVG的生成。基于预训练VLMs，我们的方法天然具备基于视觉和文本指令推理的能力，能够在从图标到复杂插画和动漫角色等多领域生成可编辑、高保真的SVG。

为推动SVG合成的发展，我们引入了MMSVG-2M，这是一个包含两百万丰富标注资产的多模态SVG合成数据集，涵盖图标、插画和动漫设计。我们还建立了标准化评测基准MMSVG-Bench，针对三大SVG生成任务："Text-to-SVG"、"Image-to-SVG"和"Character Reference SVG Generation"。大量实验表明，OmniSVG能生成高度细致和复杂的SVG内容，在定量和定性上均超越现有方法。

总结如下，我们的主要贡献包括：

* 我们提出了OmniSVG，首个利用预训练Vision-Language Models（VLMs）的端到端多模态SVG生成家族，能够生成从简单图标到复杂动漫角色的高复杂度SVG。
* 我们发布了MMSVG-2M，一个包含两百万SVG资产的大规模数据集，并配套标准化评测协议，为未来研究提供了全面资源。
* 大量实验表明，OmniSVG在定量和定性上均优于现有SVG生成方法，展现出集成到专业SVG设计工作流的潜力。

# 2. 相关工作

**图像矢量化（Image Vectorization）**。近期的矢量化进展主要利用与可微分光栅化器配合的扩散模型，如DiffVG [25]。这些方法旨在通过分数蒸馏采样（SDS）[32]，将光栅图像转换为SVG路径。具体来说，LIVE [30]采用分层优化策略，结合自交损失和无向距离引导聚焦损失；VectorFusion [19]将预训练的文本到图像扩散模型（Stable Diffusion [38]）与SVG特定正则项结合；SVGDreamer [7]引入了语义驱动的图像矢量化流程和基于粒子的SDS损失以提升视觉质量。尽管这些方法取得了显著成果，但仍存在过度平滑、颜色过饱和和可编辑性不足等局限，常常生成缠绕路径，难以捕捉专业SVG设计中固有的层次结构。

**SVG生成（SVG Generation）**。早期的SVG生成尝试构建序列模型，如RNNs [15, 16, 36, 40, 41]、VAEs [4, 28, 42, 45, 47]和Transformers [4, 52]，将SVG命令压缩为潜在表示。DeepSVG [4]进一步采用双transformer架构对SVG进行参数化，但在几何一致性上表现不佳。近年来，大语言模型（LLMs）[5, 6, 26, 48, 57–60]的出现释放了通过XML代码合成生成SVG的潜力。StrokeNUWA [44]将矢量图像转为VQVAEs编码的token序列。LLM4SVG [55]采用结构化SVG编码方法，利用可学习的语义token精确表示SVG组件及其属性。StarVector [37]结合图像编码器与LLM，从图像生成SVG代码，但该方法将文本和图像理解分离，导致文本描述与视觉输出对齐不足。然而，这些方法受限于token上下文窗口，难以处理需要长序列的复杂SVG，如插画和动漫角色常常需要高达1万甚至3万个token。

**SVG数据集与基准（SVG Datasets and Benchmarks）**。缺乏适用于复杂SVG结构的数据集是主要挑战。现有数据集[9, 20, 49]主要关注简化的基于路径的SVG或单色图标，忽略了真实设计中丰富的层次结构和色彩语义。例如，FIGR-8-SVG [9]数据集以单色图标为主，StarVector [37]提出了包含插画、图标、表情和字体的分类数据集，但其样本token长度仅至8.2k，仍未覆盖复杂层次和丰富色彩。VGBench [65]等基准进一步揭示了多格式测试和插画SVG覆盖不足的问题。

为弥补上述不足，我们引入了MMSVG-2M，这是一个包含两百万丰富标注资产的多模态SVG合成数据集，包括图标、插画和复杂动漫设计。此外，我们提出了标准化评测基准MMSVG-Bench，覆盖不同复杂度的SVG生成任务。与以往工作不同，我们的基准强调真实应用，完全公开，且在生成保真度、多样性和可编辑性等关键指标上进行严格评测。这为未来研究奠定了基础，使SVG合成更具多样性和用户导向。

# 3. MMSVG-2M

MMSVG-2M是一个大规模SVG数据集，包含两百万个SVG样本，涵盖网站图标、插画、平面设计、动漫角色等（见3.1节）。为进一步促进SVG生成方法的下游发展，我们还引入了MMSVG-Bench，这是一个由一系列多模态指令跟随任务组成的条件SVG生成基准（见3.2节）。

**数据整理（Data Curation）**。我们通过基于文件名、SVG代码和元数据的综合去重流程提取SVG样本。首先将收集到的SVG适配到$200\times200$的viewbox。然后，采用现成的VLM（如BLIP-2 [24]）为SVG生成caption。更多MMSVG-2M数据集样例见图10，指令模板见附录A.2。

**SVG简化（SVG Simplification）** 是SVG数据清洗的重要步骤，因为网络爬取的SVG数据中复杂的XML语法会导致基本形状表达的歧义。例如，Rect命令通过起点、宽度和高度参数创建矩形形状，也可以用四条正交直线表示。Circle命令基于中心点和半径创建圆形，也可用Bezier曲线近似。Transform属性则对现有形状施加仿射变换。为标准化训练和评测，我们用表2所示的原子命令简化所有SVG命令。受FIGR-8-SVG [9]和IconShop [52]启发，我们移除所有属性，并用五种基本命令（Move To (M)、Line To (L)、Cubic Bézier (C)、Elliptical Arc (A)、ClosePath (Z)）简化每个SVG。原子命令的引入进一步消除了歧义，因为复杂XML语法可用若干原子命令组合近似。为高效生成统一且简洁的数据结构，我们利用picosvg移除如group、transform等语法，并将复杂命令简化为原子路径命令。值得注意的是，原子路径命令足以表达如图1所示的复杂SVG。

## 3.1 数据整理

**数据来源（Data Source）**。我们从互联网收集的原始SVG分为三类：MMSVG-Icon来自Iconfont，SVG插画来自iconsount，SVG动漫角色来自Freepik及我们的数据生成流程（见图2）。这些网站均为用户可发布和分享SVG的在线平台，涵盖丰富类别。具体来说，我们的数据集包含200万个SVG样本，分为110万个SVG图标、50万个SVG插画和40万个SVG动漫角色。表1给出了MMSVG-2M的数据统计。

表1. MMSVG-2M数据统计。MMSVG-2M包含110万个SVG图标、50万个SVG插画和40万个SVG动漫角色。

| 数据集              | 训练集 | 验证集  | 测试集 | 来源                  | Token长度      |
|---------------------|--------|--------|-------|---------------------|-----------------|
| MMSVG-Icon          | 99万   | 10.67万| 3.3千 | Iconfont            | 2.2k ±0.9k      |
| MMSVG-Illustration  | 45万   | 4.85万 | 1.5千 | IconSount           | 8.1k ± 3.3k     |
| MMSVG-Character     | 35万   | 4.89万 | 1.1千 | Freepik & 生成      | 28k ± 7.3k      |

表2. SVG绘图命令。列出了本工作中使用的绘图命令及其参数和可视化。起始位置$(x_1,y_1)$隐式定义为前一命令的终点。

| 命令         | 参数                     | 描述                                                         | 可视化         |
|--------------|--------------------------|--------------------------------------------------------------|----------------|
| <SOP>        | ∅                        | 路径起始token                                               |                |
| M (MoveTo)   | x2,y2                    | 移动到(x2,y2)但不绘制                                       | <image src="https://arxiv.org/html/2504.06263v1/x3.png" width="100"> |
| L (LineTo)   | x2,y2                    | 绘制到(x2,y2)的直线                                         | <image src="https://arxiv.org/html/2504.06263v1/x4.png" width="100"> |
| C (Cubic Bézier) | qx1,qy1 qx2,qy2 x2,y2 | 用控制点(qx1,qy1),(qx2,qy2)和终点(x2,y2)绘制三次Bezier曲线 | <image src="https://arxiv.org/html/2504.06263v1/x5.png" width="100"> |
| A (Elliptical Arc) | rx,ry φ x2,y2        | 用半径rx,ry、旋转角φ和终点(x2,y2)绘制椭圆弧                 | <image src="https://arxiv.org/html/2504.06263v1/x6.png" width="100"> |
| Z (ClosePath)|                          | 闭合当前路径                                                 | <image src="https://arxiv.org/html/2504.06263v1/x7.png" width="100"> |
| F (Fill)     | fill                     | 设置路径填充颜色                                             |                |
| <EOS>        |                          | SVG结束token                                                |                |

![](https://cdn-mineru.openxlab.org.cn/extract/bd462ee4-da41-4d04-8103-15f5a4059566/04cef8e151bc1cbb6b8cbb582c316c27763e31ce43b9f93638382507bc664d21.jpg)
图2. OmniSVG概览。OmniSVG基于预训练视觉语言模型Qwen2.5-VL，并集成SVG tokenizer。模型将文本和图像输入分别编码为前缀token，SVG tokenizer将矢量图形命令编码到统一表示空间。

## 3.2 多模态指令跟随SVG生成

MMSVG-Bench主要关注以下SVG生成任务：

- **Text-to-SVG**：要求模型根据文本指令生成SVG。包括MMSVG-2M数据集的测试集（每张图像配有文本描述）和FIGR-8-SVG数据集，支持从文本生成更简单（路径更少）的图标。评测指标包括Frechet Inception Distance (FID) [46]、文本-SVG对齐的CLIP分数[34]、美学分数[39]和HPS（Human Preference Scores）[54]。
- **Image-to-SVG**：评估模型将图像转换为SVG的能力。MMSVG-Icon、MMSVG-Illustration和MMSVG-Character提供递增的视觉复杂度。我们为这些数据集提供训练、验证和测试划分。评测指标包括DinoScore (DINO) [31]、DinoV2特征余弦相似度、结构相似性指数(SSIM)[50]、感知图像块相似度(LPIPS)[61]和均方误差(MSE)。
- **Character-Reference SVG Generation**：评估模型在保持输入图像角色特征的同时生成新SVG的能力。与Image-to-SVG不同，Character-Reference SVG Generation不是重建输入图像的SVG，而是为输入图像重新创作特定角色SVG（见图6）。由于缺乏现有基线，我们采用第三方VLM助手GPT-4o [18]对输入角色图像与生成SVG的对齐度进行1-10分打分，1分表示差异大，10分表示高度一致。

评测时，我们还统计生成SVG样本的平均token长度（用Qwen2.5-VL [1] tokenizer），并计算每个SVG样本的平均生成时间以衡量计算成本。 

# 4. OmniSVG

为支持多模态SVG生成的端到端训练，OmniSVG将表示SVG的一系列原子路径命令参数化为序列，然后与多模态指令一起输入到预训练VLM中。

**SVG参数化（SVG Parameterization）**。如第3章所示，我们的MMSVG-2M数据集通过移除所有属性，仅保留五种基本命令（Move To (M)、Line To (L)、Cubic Bézier (C)、Elliptical Arc (A)、ClosePath (Z)）来简化每个SVG。简化后，一个SVG脚本$G$可表示为$M$条路径的组合，$G=\{P_{i}\}_{i=1}^{M}$。其中，$P_{i}$为第$i$条路径，包含$N_{i}$个命令，$P_{i}=\{C_{i}^{j}\}_{j=1}^{N_{i}}$，$C_{i}^{j}$为第$i$条路径的第$j$个命令。每个命令表示为$C_{i}^{j}=(U_{i}^{j},V_{i}^{j})^{*}$，包含类型标识$U_{i}^{j}\in\{M,L,C,A,Z\}$及对应位置参数$V_{i}^{j}$。M命令表示将光标移动到终点$(x_2,y_2)$但不绘制，L表示绘制到$(x_2,y_2)$的直线，C表示用控制点$(q_{x1},q_{y1})$、$(q_{x2},q_{y2})$和终点$(x_{2},y_{2})$绘制三次Bezier曲线，A表示用半径$r_x$、$r_y$、旋转角$\varphi$和终点$(x_2,y_2)$绘制椭圆弧，Z表示闭合路径。命令可视化见表2。然而，这些命令仅用于绘制线条，不含颜色。为补足SVG填充属性，我们将SVG Fill (F)属性的十六进制形式分配独特编号，与原始SVG命令和坐标区分。该命令用于填充每条路径的颜色。最终，我们可用$U_{i}^{j}\in\{M,L,C,A,Z,F\}$六种命令类型参数化彩色SVG。

**模型架构（Model Architecture）**。OmniSVG基于Qwen2.5-VL架构，这是一种视觉-语言模型（VLM），擅长处理视觉和文本输入，能够生成精确紧凑的SVG输出。OmniSVG可用于Text-to-SVG、Image-to-SVG和Character-Reference SVG生成任务。

如图2所示，我们首先将交错的文本和图像输入编码为前缀token，然后用SVG tokenizer将SVG脚本编码为序列，并拼接到前缀token后。完整序列作为decoder-only语言模型的输入。

具体来说，SVG tokenizer将SVG脚本$X_s$转为有序SVG token序列，嵌入到与预训练VLM相同的表示空间。借鉴IconShop [52]，我们将SVG脚本的层次结构展平，将不同路径拼接为单一命令序列，每条路径以绘图命令和点坐标开始。因此，每个SVG序列可展平成一维序列。作为生成标识，我们在SVG序列两端加上特殊token如${\mathrm{\overline{{<SOP>}}}}$和$<\tt EOS>$，分别标识SVG序列的起止。每种命令类型（M,L,C,A,Z,F）分配特殊token。为缩短SVG序列长度，我们将点坐标合并为一个token，映射函数为$<x,y>\to x\times w+y$，$w$为图像宽度。SVG序列通过可学习嵌入层提升到与预训练VLM相同的嵌入空间。

![](https://cdn-mineru.openxlab.org.cn/extract/bd462ee4-da41-4d04-8103-15f5a4059566/ad9bc5619ecda4dc549241db7ffb5c9a9591f7fc3acba83b5ad84e018bfe64f6.jpg)
图3. OmniSVG模型的训练与验证困惑度（PPL）。所有模型均在2500亿token上从头训练。可见模型规模越大，性能越好。

**训练目标（Training Objective）**。类似LLMs，我们训练模型在给定前缀token条件下，用下一个token预测损失生成新token。

$$
\theta^{*}=\arg\operatorname*{max}_{\theta}\prod_{i=1}^{L}P\left(x_{s,i}\mid x_{s,<i},x_{c}\right)
$$

**模型扩展（Scaling Up）**。为研究多模态SVG生成的扩展效果，我们将OmniSVG从3B参数扩展到7B参数。图3展示了两种模型在2500亿token上从头训练的困惑度。结果表明，模型规模越大，验证困惑度越低，生成验证数据的概率越高。 

# 5. 实验

为验证我们方法的有效性，我们首先介绍基线方法和实现细节（见5.1节）。随后与现有方法进行定量对比（见5.2节），并通过消融实验（见5.3节）分析设计的有效性。

## 5.1 基线方法

在Text-to-SVG任务中，我们与基于语言（LLM-based）的方法进行对比，包括VectorFusion [19]、SVGDreamer [56]、Chat2SVG [53]和IconShop [52]。在Image-to-SVG任务中，我们与图像矢量化和多模态大语言模型方法进行对比，包括LIVE [30]、DiffVG [25]、StarVector [37]和GPT-4o [18]，均采用作者公开实现和推荐超参数，并按需应用其预处理和后处理代码。

具体来说，Text-to-SVG任务中，优化型方法SVGDreamer通过语义驱动的图像矢量化流程有效分离前景与背景，提升了可编辑性，但难以处理复杂场景。另一优化型方法VectorFusion无需大规模带标注SVG数据集即可生成可导出的SVG矢量图，但同样难以应对复杂场景和多样风格。这类优化型方法的主要问题是优化时间过长，生成一个SVG通常需十分钟以上，计算开销过大。LLM-based方法中，Chat2SVG集成了大语言模型与图像扩散模型生成语义丰富的SVG模板，但仍需对LLM输出的SVG脚本进行优化，增加了计算复杂度并带来训练难题。相比之下，IconShop采用基于transformer的架构自回归建模SVG路径序列，在简化图标SVG场景下表现优异，是Text-to-SVG生成的有效方案，但仅能生成黑色简单图标SVG。

在Image-to-SVG任务中，我们与图像矢量化方法对比。LIVE支持渐进高效地生成SVG，在光栅图像监督下优化封闭矢量路径并可控形状复杂度，但生成复杂SVG时优化耗时极长。DiffVG实现了矢量图形光栅化的端到端可微分性，通过抗锯齿和基于梯度的方法提升优化效果，但因前后向光栅化过程复杂，计算开销大。近期多模态大语言模型（MLLM）方法StarVector将视觉理解与LLM架构结合，可从文本和图像输入生成SVG，但仍难以生成复杂SVG。由于StarVector [37]尚未公开其Text-to-SVG模型权重，MMSVG-Bench未评测其Text-to-SVG能力。MMSVG-Bench还用VLM方法GPT-4o对我们方法进行综合评测。我们在MMSVG-2M数据集上与这些基线方法对比，涵盖简单的MMSVG-Icon、较复杂的MMSVG-Illustration和极复杂的MMSVG-Character子集。定量结果见表3，定性结果见图4和图5。

![](https://cdn-mineru.openxlab.org.cn/extract/bd462ee4-da41-4d04-8103-15f5a4059566/47b8487af259b4044e8c8dba25d8851cadc55a44f7ed5493606f6a74de8bec48.jpg)
图4. Text-to-SVG任务与SOTA方法的定性对比。我们在Icon、Illustration和Character基准上与SOTA方法对比，所提方法在指令遵循和生成SVG美学质量上均优于现有方法。

## 5.2 评测与对比

### 5.2.1 定量评测

我们在MMSVG-2M数据集（含MMSVG-Icon、MMSVG-Illustration和MMSVG-Character）上报告OmniSVG与基线方法的评测指标，涵盖Text-to-SVG和Image-to-SVG任务。表3显示，OmniSVG在所有Text-to-SVG评测指标上均优于其他基线，表现为更低的FID、更高的CLIP分数、更高的美学分数和HPS分数。Image-to-SVG任务中，虽然图像矢量化方法LIVE在SSIM、LPIPS和MSE上表现更优，但我们的OmniSVG在DINO分数上更高，表明输入图像与生成SVG的语义特征更一致。

![](https://cdn-mineru.openxlab.org.cn/extract/bd462ee4-da41-4d04-8103-15f5a4059566/c4d1bae60a0f7e699e437b9eff0080a014ccb4cd6a0bdf64ba532053813994d9.jpg)
图5. Image-to-SVG任务与SOTA方法的定性对比。我们在基准上与SOTA方法对比。优化型方法如DiffVG [25]和LIVE [30]在复杂图像上易出现伪影，GPT-4o [18]即使输入复杂图像也只能生成图标级SVG，StarVector [37]能处理图标但难以应对插画和复杂角色。

### 5.2.2 定性评测

Text-to-SVG任务如图5所示，我们用7个不同文本指令对比多种基线方法。优化型方法如SVGDreamer [56]和VectorFusion [19]生成单个SVG需大量计算，依赖迭代优化，虽能细化细节但耗时高。语言模型方法如IconShop [52]和Chat2SVG [53]生成速度快，因可直接利用预训练模型生成SVG模板，但各有局限。IconShop虽能生成完整SVG形状，但仅限黑白图形，难以满足彩色和丰富视觉需求。Chat2SVG为两阶段流程，此处仅关注其第一阶段生成初始SVG模板，虽有一定灵活性，但细节和语义一致性不及我们方法。

![](https://cdn-mineru.openxlab.org.cn/extract/bd462ee4-da41-4d04-8103-15f5a4059566/9f1ea869dbbc5598d7a3ec9bea6be9461c0cfe5f8ee0b2e92bb2d40cc1394f4e.jpg)
图6. OmniSVG生成的角色参考SVG（CRef）。通过在MMSVG-Character上用自然角色图像与SVG配对数据训练，OmniSVG可通过图像参考生成角色SVG。

我们的方法OmniSVG在各种文本指令下均优于其他基线，生成SVG不仅高度还原输入指令，还具备丰富色彩和几何准确性，并能处理更复杂视觉线索，支持简单与复杂SVG设计。该定性评测显示我们方法在速度和质量上均优于现有方法，是更高效多样的SVG生成方案。

Image-to-SVG任务中，我们与DiffVG [25]、LIVE [30]、GPT-4o [18]和StarVector [37]等经典方法对比。结果显示我们方法在SVG生成质量和效率上均优于这些基线。

我们对比了优化型（DiffVG、LIVE）和神经网络型（GPT-4o、StarVector）SOTA方法。DiffVG和LIVE在简单图标上表现良好，但面对复杂图像时易生成伪影，难以保持复杂结构的完整性。GPT-4o虽能处理复杂输入，但输出仅限图标级SVG，说明其设计或训练数据限制了复杂插画的理解与转换。StarVector在图标任务表现强，但对插画和复杂角色输入时难以生成准确SVG，泛化能力有限。

而我们的OmniSVG能高效将图标、插画、复杂角色等多样输入图像转为高质量、可编辑SVG，展现出对复杂视觉线索的优越处理能力，区别于传统矢量化方法。更多视觉结果见图8。

Character-Reference SVG生成任务如图6所示，通过在MMSVG-Character上用自然角色图像与SVG配对数据训练，OmniSVG可通过图像参考生成角色SVG。我们还用第三方VLM助手GPT-4o [18]评估输入角色图像与生成SVG的对齐度，平均得分约为7，表明生成的IP SVG与参考图像高度相似。

![](https://cdn-mineru.openxlab.org.cn/extract/bd462ee4-da41-4d04-8103-15f5a4059566/8f1ea869dbbc5598d7a3ec9bea6be9461c0cfe5f8ee0b2e92bb2d40cc1394f4e.jpg)
图8. OmniSVG生成的角色参考SVG（CRef）。通过在MMSVG-Character上用自然角色图像与SVG配对数据训练，OmniSVG可通过图像参考生成角色SVG。

### 5.3 消融实验

我们对SVG参数化、模型规模扩展和VLM架构进行了消融实验，以验证设计的有效性。大部分消融实验在OmniSVG-3B模型上进行以加快迭代，实验证明这些设计在更大的OmniSVG-7B上同样有效。

**SVG参数化有效性**。本实验对比了SVG参数化方法与传统非参数化方法在大语言模型中的表现。两者的根本区别在于坐标信息的token化方式：传统方法将每个数字作为单独token（如坐标(123, 456)需6个token加分隔符），而我们的参数化方法将整个坐标编码为单一语义token，大幅提升模型处理复杂路径信息的能力。

我们对SVG的坐标和颜色属性均进行了参数化。为验证设计有效性，进行了如下消融实验：仅参数化颜色、不参数化坐标；仅参数化坐标、不参数化颜色；两者均不参数化；两者均参数化（本方法）。表4和图7总结了在MMSVG-Illustration数据集上的实验结果。

表4. SVG参数化定量消融。对比了颜色参数化（color param.）和坐标参数化（coord param.）的影响。

| 方法                | Text-to-SVG |        |        |      | Image-to-SVG |        |        |      | # Tokens |
|---------------------|-------------|--------|--------|------|--------------|--------|--------|------|----------|
|                     | FID↓        | CLIP↑  |        | HPS↑ | DINO↑        | SSIM↑  | LPIPS↓ | MSE↓ |          |
| 无参数化             | 97.01       | 0.1537 |        |2.873 | 0.112        | 0.702  | 0.668  |0.334 | 35k      |
| 无坐标参数化          | 87.93       | 0.2193 |        |3.352 | 0.156        | 0.783  | 0.769  |0.295 | 29.5k    |
| 无颜色参数化          | 76.39       | 0.2632 |        |4.234 | 0.201        | 0.901  | 0.912  |0.098 | 13.1k    |
| 全参数化（本方法）     | 70.45       | 0.3077 |        |5.39  | 0.245        | 0.974  | 0.944  |0.069 | 9.7k     |

表4结果显示，同时参数化坐标和颜色在所有评测指标上表现最佳，且token长度最短，说明本方法高效利用token空间，能以更低计算资源生成复杂SVG，适合大规模和高效应用。图7定性结果也表明，双参数化方法在SVG复杂度提升时显著优于其他方法，无参数化方法在复杂图像下甚至无法生成SVG。实验验证了本方法在性能与资源利用间的优越平衡。

**模型规模消融**。为分析大模型是否有助于SVG生成，我们在MMSVG-2M数据集上评测了不同规模的OmniSVG基础模型，见表5。模型规模从FLAN-T5-base [8]（223M）逐步扩展到Qwen2.5-VL-3BInstruct（3.7B）和Qwen2.5-VL-7B-Instruct（8.3B）。结果显示，模型规模越大，生成样本质量越高，验证了自回归大模型在多模态SVG生成任务上的扩展有效性。

表5. 模型规模消融。模型规模越大，生成样本质量越高。

| 方法                   | 输入      | 规模  | FID↓  | CLIP↑ | HPS↑ | DINO↑ | SSIM↑ | LPIPS↓ | MSE↓  |
|-----------------------|----------|-------|-------|-------|------|-------|-------|--------|-------|
| FLAN-T5-Base[8]       | Text     | 223M  | 89.30 | 0.177 |3.437 | 0.102 | -     | -      | -     |
| FLAN-T5-Large[8]      | Text     | 770M  | 83.92 |0.2303 |4.102 | 0.177 | -     | -      | -     |
| FLAN-T5-xl[8]         | Text     | 3B    | 76.92 |0.2789 |4.642 | 0.217 | -     | -      | -     |
| blip2-flan-t5-xl[24]  | Text/Image|3.94B | 75.23 |0.2881 |4.712 | 0.201 |0.891  |0.902   |0.046  |
| OmniSVG(3B)           | Text/Image|3.7B  | 70.45 |0.3077 |5.395 | 0.245 |0.974  |0.944   |0.019  |
| OmniSVG(7B)           | Text/Image|8.3B  | 66.91 |0.3164 |5.59  | 0.253 |0.988  |0.959   |0.013  |

**VLM架构消融**。为验证VLM架构的有效性，我们用CLIP ViT-B/32 [33]、VQGAN [12]和本方法采用的Qwen2.5-VL [1]替换VLM架构，在MMSVG-Illustration数据集上对比不同架构的SVG生成表现。结果见表6，Qwen2.5-VL在所有指标上均表现最佳，且支持多图输入，能生成更复杂的输出（如角色参考SVG）。

表6. VLM架构消融。Qwen2.5-VL在所有指标上表现最佳。

| Vision   | Language   | FID↓   | CLIP↑  | HPS↑  | DINO↑ | SSIM↑ | LPIPS↓ | MSE↓  |
|----------|------------|--------|--------|-------|-------|-------|--------|-------|
| CLIP     | Qwen2.5    | 72.243 |0.2876  | 4.57  |0.209  |0.881  |0.872   |0.176  |0.062  |
| VQGAN    | Qwen2.5    | 76.29  |0.2739  | 4.62  |0.198  |0.890  |0.852   |0.183  |0.065  |
| Qwen2.5-VL-3B-Instruct|        | 70.45  |0.3077 | 5.39  |0.245  |0.974   |0.944  |0.069  |0.019  |
| Qwen2.5-VL-7B-Instruct|        | 66.91  |0.3164 | 5.59  |0.253  |0.988   |0.959  |0.041  |0.013  |

![](https://cdn-mineru.openxlab.org.cn/extract/bd462ee4-da41-4d04-8103-15f5a4059566/4bf04d9307b178caf086f3b28bdd9ad98a56ec11c291ecb70f9e943534675b88.jpg)
图8. OmniSVG的SVG生成能力示例。

### 5.4 用户研究与实现细节

**用户研究（User Study）**。我们评估了用户对生成SVG的偏好、生动性以及文本/图像与SVG的对齐度。具体做法为：从评测集随机采样5个文本描述和3个图像提示，分别用我们的方法和基线模型生成SVG，并与原始8个SVG一同组成64个SVG供用户评测。表7中preference为用户偏好，Vividity为生动性，Alignment为输入与SVG的对齐度。结果显示，OmniSVG在用户偏好、生动性和对齐度上均获最高分。

表7. 用户研究结果。OmniSVG在用户偏好等方面领先。

| 方法                | Preference↑ | Vividity↑ | Alignment↑ |
|---------------------|-------------|-----------|------------|
| Vectorfusion [19]   | 40          | 60        | 80         |
| SVGDreamer [56]     | 40          | 70        | 80         |
| Chat2SVG [53]       | 60          | 60        | 90         |
| IconShop [52]       | 80          | 60        | 80         |
| GPT-4o [18]         | 40          | 60        | 80         |
| StarVector(8B) [37] | 40          | 80        | 70         |
| DiffVG[25]          | 90          | 80        | 100        |
| LIVE [30]           | 90          | 70        | 100        |
| OmniSVG[30]         | 100         | 90        | 100        |

![](https://cdn-mineru.openxlab.org.cn/extract/bd462ee4-da41-4d04-8103-15f5a4059566/13e074ff4cabd2c166f47fc9bb00a5a3fd54ba612e4a1f3c71f6fe21a1677b79.jpg)
图9. OmniSVG在Image-to-SVG任务上的局限。OmniSVG能成功生成矢量风格图像，但难以拟合自然图像。

**实现细节（Implementation Details）**。我们用bfloat16和ZeRO-2策略[35]进行高效训练，优化器采用AdamW [29]，学习率从$3\times10^{-4}$衰减到$3\times10^{-6}$，权重衰减0.1。预训练权重加载自Qwen2.5-VL [1]，SVG嵌入层随机初始化。默认用top-k（k=50）和top-p（p=0.95）采样生成SVG以保证多样性。

# 6. 结论

我们提出了OmniSVG，一个统一的多模态SVG生成框架，利用预训练视觉-语言模型（VLMs）。通过将SVG命令和坐标参数化为离散token，OmniSVG高效解耦结构逻辑与几何信息，解决了"坐标幻觉"等问题，同时保持设计表达力。我们的方法在质量和效率上均优于现有方法，能在多种设计领域生成高质量、可编辑SVG。此外，我们还提出了MMSVG-2M大规模多模态数据集（含两百万标注SVG资产）及标准化评测协议。大量实验表明，OmniSVG在多种条件生成任务上均优于现有SVG生成方法，展现出集成到专业SVG设计工作流的潜力。

**局限与未来工作**。推理时，OmniSVG在处理复杂样本时需生成数万个token，导致生成时间较长。OmniSVG仅适用于矢量风格图像提示，难以处理自然图像（见图9）。未来可探索多token预测[2, 13]和KV-cache压缩[3, 63]以降低生成成本。此外，OmniSVG的自回归特性也为in-context learning[43, 62, 64]、chain-of-thought reasoning[14, 51]和多轮交互生成[17, 27]等方向提供了机会，提升用户可控性。

# 附录

# A. MMSVG-2M数据集的更多细节

## A.1 MMSVG-2M数据集样例

我们在图10中可视化了MMSVG-2M数据集的样例。在MMSVG-2M数据集中，55%的SVG样本属于MMSVG-Icon，25%属于MMSVG-Illustration，其余20%属于MMSVG-Character。在MMSVG-Character类别中，一半样本来自Freepik，另一半由我们的数据生成流程生成。在生成过程中，我们还收集了用于角色参考SVG生成任务的图像-SVG对。

## A.2 SVG-图像-文本对的构建

我们的MMSVG-2M数据集包含两百万个SVG样本及其对应的光栅化图像。我们用BLIP-2 [24]为光栅化图像生成caption，从而为模型微调提供文本描述，使其能遵循这些指令。光栅化采用CairoSVG [21]，并移除了生成全白图像的样本。

**标注（Annotation）**。我们采用现成的VLM（如BLIP-2 [24]）用如下提示生成SVG caption。为减少幻觉，去除了CLIP分数低于30的样本。我们还用词云（见图12）可视化了MMSVG-2M数据集的标注关键词分布。不同任务的指令模板见表8。

**指令模板（Instruction templates）**。MMSVG-Bench提供三类任务：Text-to-SVG、Image-to-SVG和Character-Reference SVG Generation。每类任务需不同的指令模板。对于文本和图像条件SVG生成，我们分别输入文本或图像。对于角色参考SVG生成，我们输入两张图像，第二张为第一张的角色参考图像，生成基于这两张图像的角色SVG。不同任务的指令模板见表8。

## A.3 角色-SVG对的构建

如图1所示，MMSVG-2M-Character子集部分通过生成式流程构建。流程如图2所示，我们采用基于FLUX [22]的生成模型，并用vector-style LoRA增强以生成SVG风格数据。图像条件采用FLUX-Redux [23]，通过SigLIP编码器注入图像特征并投影为图像embedding。这些embedding与文本token拼接作为FLUX [22]的条件输入。实践中，原始Redux [23]条件过强，为此我们采用社区实现的Redux变体，在2D空间下对图像embedding降采样。实验（见图11）表明，降采样因子2~3倍时生成的SVG风格角色最合理。最后用VTracer [10]对生成图像进行快速矢量化。MMSVG-2M-Character子集构建流程为：首先从Danbooru [11]数据集中筛选出10.3万个角色实例，按上述流程生成，比较原始FLUX [22]输出与矢量化结果，仅保留PSNR和SSIM分数高于阈值的样本作为有效数据。

# B. 各基线方法更多细节

## B.1 Text-to-SVG任务

SVGDreamer [56]采用语义驱动的图像矢量化（SIVE）流程分离前景与背景，提升可编辑性。SIVE流程利用基于注意力的图元控制和注意力掩码损失，有效操控各元素。为解决现有Text-to-SVG方法的问题，提出的Vectorized Particle-based Score Distillation（VPSD）方法将SVG建模为控制点和颜色的分布，提升了形状、颜色多样性和收敛速度。

VectorFusion [19]利用在像素表示上训练的文本条件扩散模型生成可导出的SVG矢量图，无需大规模带标注SVG数据集。通过优化可微分矢量图形光栅化器，从预训练扩散模型中蒸馏语义知识，并用Score Distillation Sampling生成与caption一致的SVG。实验表明，VectorFusion提升了质量和保真度，支持像素艺术、素描等多种风格。

Chat2SVG [53]提出了融合大语言模型（LLMs）和图像扩散模型的Text-to-SVG生成混合框架。该方法首先用LLM生成语义丰富的SVG模板，再通过图像扩散模型引导的双阶段优化流程，在潜空间细化路径并调整点坐标，提升几何复杂度。

IconShop [52]采用基于transformer的架构对路径命令编码，自回归建模SVG路径序列。在简化图标场景下表现优异，为Text-to-SVG生成提供了有效方案，并通过为FIGR-8-SVG数据集添加caption扩展了数据。我们获得了其数据集和原始划分，并在该数据上用预训练checkpoint（OmniVG数据集训练）训练了我们的模型。IconShop的结果由原作者提供，用于对比。

LLM4SVG [55]是利用大语言模型（LLMs）理解和生成SVG的框架。其采用结构化SVG编码，利用可学习语义token精确表示SVG组件及属性，使LLMs能生成语义与文本描述一致、视觉连贯的SVG。但LLM4SVG最大token长度为2048，难以生成需要长序列的复杂SVG。

## B.2 Image-to-SVG任务

LIVE（Layer-wise Image Vectorization）[30]是一种递进式生成SVG的方法，通过递归添加和优化封闭矢量路径，使SVG与给定光栅图像高度拟合。其基于可微分渲染器（基于DiffVG [25]），在光栅图像监督下直接优化路径，并通过调整路径段数控制形状复杂度。引入了组件级路径初始化，识别关键视觉组件，提升拓扑提取效率并减少冗余形状。

DiffVG [25]是矢量图形研究的里程碑，首次提出可微分矢量图形光栅化管线。通过抗锯齿和梯度优化，DiffVG实现了可微分性。与依赖非可微曲线到网格转换的方法不同，DiffVG采用前向-反向光栅化流程，前向生成抗锯齿图像，反向计算矢量参数梯度。

StarVector [37]直接在SVG代码空间工作，将视觉理解与LLM架构结合，能用视觉输入生成精确SVG原语。其采用transformer架构，将图像编码器与语言模型集成，支持视觉输入到SVG代码的生成。StarVector能处理多种SVG类型（图标、logo、复杂图表），在多种矢量化任务上表现出强泛化能力。但其上下文窗口为16k token，处理极复杂SVG（需更长序列）时仍有困难。

# C. SVG Tokenizer 实现细节补充（来自官方解答整理）

根据OmniSVG作者在GitHub issues中的权威解答，SVG tokenizer的实现可分为以下几个关键阶段：

1. **多模态前缀构建（Multi-modal Prefix Construction）**
   - 首先将交错的文本和图像输入（如文本提示、参考图片）进行tokenize和embedding，作为前缀token。这些前缀token为后续SVG生成提供上下文锚点。

2. **结构化SVG编码（Structural SVG Encoding）**
   - 原始SVG脚本会被解析为原子组件（如M、L、C、A、Z、F等命令），并通过专用SVG tokenizer顺序分词。该过程将SVG基本图元转为离散token，同时保留其层级与绘制顺序。
   - 路径命令的详细定义可参考论文表2。

3. **序列组合与建模（Sequence Composition & Modeling）**
   - 前缀token与SVG token拼接为统一序列，作为decoder-only VLM架构的输入。
   - 这种顺序不仅保证了SVG渲染的z-order（后绘制的路径会遮挡前面的路径），还确保了填充命令依赖于前面的路径定义。作者实验证明，保持自然绘制顺序有助于模型性能，随意打乱顺序会破坏SVG的层次和依赖关系。

4. **坐标与颜色参数化（Coordinate & Color Parameterization）**
   - **坐标编码**：以200×200画布为例，(x, y)坐标被离散为整数（0≤x, y≤200），并通过 position_token = 200 * x + y 公式编码为单一token，总共4万种空间token。这种方式既保证了空间精度，又控制了词表规模。
   - **颜色编码**：每个RGB通道量化为4bit（16级，0-15），总共16³=4096种颜色token，极大压缩了颜色空间，兼顾表达力与词表可控性。
   - **椭圆弧参数**：如A命令涉及5个参数，均采用类似离散化和映射方式，确保每个参数都能被唯一token表示。

5. **设计权衡与优势**
   - 这种tokenizer设计有效避免了"坐标幻觉"问题，使模型能高效学习几何与色彩规律。
   - 顺序化token不仅保证了SVG语义和渲染正确性，也便于模型捕捉复杂依赖。
   - 词表规模通过离散化和量化得到有效控制，兼顾了表达能力和训练效率。

> 相关官方答复详见：
> - [Issue #9: How SVG tokenization is done?](https://github.com/OmniSVG/OmniSVG/issues/9)
> - [Issue #6: Question about position and color parameterization](https://github.com/OmniSVG/OmniSVG/issues/6)

<p align="center">
    <img src="./images/INF-LLaVA.png" width="250" style="margin-bottom: 0.2;"/>
<p>

<p align="center">
        🤗 <a href="https://huggingface.co/WeihuangLin"> Models on Hugging Face</a>
<br>

# 🌋 INF-LLaVA: Dual-perspective Perception for High-Resolution Multimodal Large Language Model
This repository contains the Pytorch code and model weight of **INF-LLaVA**, a novel MLLM designed for high-resolution image perception and reasoning.

**INF-LLaVA** has the following features to process high-resolution images:
-  **Dual-perspective Cropping Module(DCM)** : Integrate both global and local perspectives when cropping high-resolution images into subimages. This enhances the model’s ability to capture detailed and contextual information.
-  **Dual-perspective Enhancement Module(DEM)** : An effective and efficient module for fusing dual-perspective features, resulting in dual-enhanced
features that significantly improve performance.
- **Strong Performance** : **INF-LLaVA** outperforms existing models on multiple benchmarks, demonstrating the effectiveness of our approach. Check out our [model zoo](#model-zoo).


# Table of Contents

- [Install](#install)
- [Train](#train)
- [Evaluate](#evaluate)
- [Model Zoo](#model-zoo)

# Install
1. Clone this repository and navigate to INF-LLaVA folder
```bash
git clone https://github.com/WeihuangLin/INF-LLaVA.git
cd INF-LLaVA
```
2. Install Package
```Shell
conda create -n inf-llava python=3.10 -y
conda activate inf-llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```
3. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation --no-cache-dir
```

# Train

1. Pre-train
```bash
cd INF-LLaVA
bash INF-LLava_pretrain.sh
```
**Note:** You should replace the data_path and image_folder in the `INF-LLava_pretrain.sh`

2. Finetune
```bash
cd INF-LLaVA
bash INF-LLava_finetune.sh
```
**Note:** You should replace the data_path and image_folder in the `INF-LLava_finetune.sh`

You can download our pretrained weights in [Model Zoo](#model-zoo) 

# Evaluate

We follow [lmm-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval/) to conduct evaluations. Please refer to [lmm-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval/) for help. We provide the same script to complete the testing.


# Model Zoo

| Version |  Checkpoint |
|----------|----------|
| $INF-LLaVA$ | 🤗[WeihuangLin/INF-LLaVA-sft](https://huggingface.co/WeihuangLin/INF-LLaVA-sft/)
| $INF-LLaVA^*$ |  🤗[WeihuangLin/INF-LLaVA_star-sft](https://huggingface.co/WeihuangLin/INF_star-LLaVA-sft/)

$INF-LLaVA^*$ means using a more diverse dataset for training.

--- 
# 🙏 Acknowledgement
We are thankful to [LLaVA](https://github.com/haotian-liu/LLaVA), [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) and [LLama3](https://github.com/meta-llama/llama3) for releasing their models and code as open-source contributions.

In case if you face any issues or have any questions, please feel free to create an issue.


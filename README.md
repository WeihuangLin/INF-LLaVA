<p align="center">
    <img src="./images/INF-LLaVA.png" width="250" style="margin-bottom: 0.2;"/>
<p>

<p align="center">
  <a href='https://www.arxiv.org/abs/2407.16198'>
    <img src='https://img.shields.io/badge/Paper-PDF-green?style=flat&logo=arXiv&logoColor=green' alt='arXiv PDF'> </a>
  
  <a href='https://huggingface.co/collections/WeihuangLin/inf-llava-669be442004e418e71fea201' style='padding-left: 0.5rem;'>
    <img src='https://img.shields.io/badge/Huggingface%20Model-8A2BE2' alt='Project Page'> </a>
</p>


# 🌋 INF-LLaVA: Dual-perspective Perception for High-Resolution Multimodal Large Language Model
This repository contains the Pytorch code and model weight of **INF-LLaVA**, a novel MLLM designed for high-resolution image perception and reasoning.

**INF-LLaVA** has the following features to process high-resolution images:
-  **Dual-perspective Cropping Module(DCM)** : Integrate both global and local perspectives when cropping high-resolution images into subimages. This enhances the model’s ability to capture detailed and contextual information.
-  **Dual-perspective Enhancement Module(DEM)** : An effective and efficient module for fusing dual-perspective features, resulting in dual-enhanced
features that significantly improve performance.
- **Strong Performance** : **INF-LLaVA** outperforms existing models on multiple benchmarks, demonstrating the effectiveness of our approach. Check out our [model zoo](#model-zoo).

## News !!
- 🔥[2024-7-19] Release the ckpt model of **INF-LLaVA** on [Hugging Face](https://huggingface.co/collections/WeihuangLin/inf-llava-669be442004e418e71fea201).
- 🔥[2024-7-16] Release the code of **INF-LLaVA**.

## To-Do Lists 

- [ ] Release **INF-LLaVA** model based on [Llama 3.1](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) 
- [x] Release **INF-LLaVA** Strong Models. 
- [x] Release **INF-LLaVA** training code. 


## Table of Contents

- [Install](#install)
- [Train](#train)
- [Evaluate](#evaluate)
- [Model Zoo](#model-zoo)

## Install
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

## Train

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

## Evaluate

We follow [lmm-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval/) to conduct evaluations. Please refer to [lmm-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval/) for help. We provide the same script to complete the testing.


## Model Zoo

| Version |  Checkpoint |
|----------|----------|
| $INF-LLaVA$ | 🤗[WeihuangLin/INF-LLaVA-sft](https://huggingface.co/WeihuangLin/INF-LLaVA-sft/)
| $INF^*-LLaVA$ |  🤗[WeihuangLin/INF_star-LLaVA-sft](https://huggingface.co/WeihuangLin/INF_star-LLaVA-sft/)

$INF^*-LLaVA$ means using a more diverse dataset for training.

## 🎫 License

This project is released under the [Apache 2.0 license](LICENSE). 

## 🖊️ Citation

If you find this project useful in your research, please consider cite:

```BibTeX


@misc{ma2024infllava,
      title={INF-LLaVA: Dual-perspective Perception for High-Resolution Multimodal Large Language Model}, 
      author={Yiwei Ma and Zhibin Wang and Xiaoshuai Sun and Weihuang Lin and Qiang Zhou and Jiayi Ji and Rongrong Ji},
      journal={arXiv preprint arXiv:2407.16198},
      year={2024}
}

```

## 🙏 Acknowledgement
We are thankful to [LLaVA](https://github.com/haotian-liu/LLaVA), [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) and [LLama3](https://github.com/meta-llama/llama3) for releasing their models and code as open-source contributions.

In case if you face any issues or have any questions, please feel free to create an issue.

# Enhanced Neural Machine Translation using Conformer-augmented Encoder

## Overview

**Neural Machine Translation (NMT)** is a deep learning-based method to translate text from one language to another. While Transformer models have achieved state-of-the-art performance in this area, they struggle with computational efficiency and local feature modeling — especially for **low-resource Indian languages**.

Our work introduces a **Conformer-based Encoder** into the NMT architecture to improve:

- Local and global dependency modeling
- Translation quality on morphologically rich, low-resource languages
- Computational efficiency

---

## Key Features & Contributions

> **This repository builds on top of OpenNMT-py. All original base code and structure belong to [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py).**

### Enhancements Added:

- ✅ Integrated a **Conformer Encoder** module
  - Combines convolution + self-attention layers
  - Supports variable kernel sizes
- ✅ Modified configuration options to support Conformer parameters
- ✅ Custom model classes for compatibility with OpenNMT training pipeline
- ✅ Evaluation and ablation study scripts tailored for Hindi-English NMT
- ✅ Trained models and BLEU score comparisons included

## Setup & Usage

### Hugging Face Token Required

To download pretrained models hosted on Hugging Face, generate a token [here](https://huggingface.co/settings/tokens) and login:

```bash
huggingface-cli login
```

### 1. Clone the Repository

```bash
git clone https://github.com/anandtharad/AksharNMT.git
cd AksharNMT
```

### 2. Install Dependencies

```bash
pip install -e .
```

### 3. Execute

To reproduce our results end-to-end, use:

```bash
bash run.sh
```

### 4. Gradio Interface

To directly use the pretrained models for inference

```bash
python gradio-testing-app/app.py
```

## Authors

- Anand Tharad
- Animesh Pradhan
- Sanjana Pradhan
- Sumit Raj

## Acknowledgements

Original implementation: [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py)

Conformer inspiration: `Gulati et al., "Conformer: Convolution-augmented Transformer"`

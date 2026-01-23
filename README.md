# 🚀 Elastic Attention

## 📖 Introduction


Within only 12 hours of training on 8$\times$A800 GPUs, **Elastic Attention**  enables models to achieve both strong performance and efficient inference.
Experiments across 3 long-context scenarios with Llama-3.1 and Qwen3-series models exhibit the superiority of our method.

## 💻 System Environment

The codebase has been strictly verified on the following high-performance computing environment:

| Component | Specification | Notes |
| --- | --- | --- |
| **OS** | Ubuntu 22.04.4 LTS | Tested on ID: `ubuntu` |
| **Python** | 3.11+ | Recommended |
| **PyTorch** | 2.6.0 | Ecosystem compatible |
| **CUDA** | 12.4+ | **Required** |
| **GPU** | NVIDIA A100/H100 (80GB) | High VRAM required |

## ⚙️ Installation

### 1. Setup Python Environment

First, clone the repository and set up the basic PyTorch ecosystem.

```bash
# Clone the repository
git clone https://github.com/LCM-Lab/Elastic-Attention.git
cd Elastic-Attention

# Install PyTorch ecosystem (CUDA 12.4)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

```

### 2. Install Dependencies

This project relies on `Block-Sparse-Attention` and other libraries.

```bash
# 2.1 Install Block-Sparse-Attention (Custom CUDA Ops)
git clone https://github.com/mit-han-lab/Block-Sparse-Attention
cd Block-Sparse-Attention

# Ensure CUDA_HOME matches your local path (adjust if necessary)
export CUDA_HOME=/usr/local/cuda-12.4/
python setup.py install
cd ..

# 2.2 Install other python dependencies
pip install -r requirements.txt
pip install modelscope  # Required for data download

```

### 3. Install Elastic Attention

Finally, install the main package in editable mode.

```bash
pip install -e .

```

## 📚 Data Preparation

We provide scripts to download the Supervised Fine-Tuning (SFT) datasets via [ModelScope](https://www.google.com/search?q=https://modelscope.cn/).

### Download SFT Datasets

You can use the following Python snippets to download the processed data:

**Qwen Mix SFT (64K)**

```python
from modelscope.msdatasets import MsDataset
ds = MsDataset.load('LCM_group/qwen_mix_sft_64K6', subset_name='default', split='train')

```

**LLaMA Mix SFT (64K)**

```python
from modelscope.msdatasets import MsDataset
ds = MsDataset.load('LCM_group/llama_mix_sft_64K6', subset_name='default', split='train')

```

> **Note:** For the training demo below, we include a small cached dataset located at `sparseattn/public_data/data_cache/demo_data_qwen_packed_maxseq65536.parquet`.

## 🏰 Model Zoo

Pre-trained models and checkpoints are available on ModelScope.

| Model Series | Model Scale | Link |
| --- | --- | --- |
| **Elastic-Attention Collection** | 4B/8B | [ModelScope Collection](https://modelscope.cn/collections/LCM_group/Elastic-Attention) |

## 🏃 Training

To start training with the provided demo data, utilize the included startup script.

```bash
# Grant execution permissions
chmod +x sparseattn/run_scripts/training.sh

# Run the training script
bash sparseattn/run_scripts/training.sh

```

*Configuration details (batch size, learning rate, etc.) can be modified inside `sparseattn/run_scripts/training.sh`.*

## ⚖️ Evaluation

We recommend using **LOOM-Eval** for comprehensive evaluation of long-context capabilities.

* **Repository:** [LCM-Lab/LOOM-Eval](https://github.com/LCM-Lab/LOOM-Eval)

Please refer to the LOOM-Eval repository for detailed instructions on how to evaluate the checkpoints produced by Elastic Attention.

## 🔗 Related Implementations

We acknowledge and reference the following open-source implementations:

| Method | Repository |
| --- | --- |
| **NSA** (Native Sparse Attention) | [XunhaoLai/native-sparse-attention-triton](https://www.google.com/search?q=https://github.com/XunhaoLai/native-sparse-attention-triton) |
| **MoBA** | [MoonshotAI/MoBA](https://www.google.com/search?q=https://github.com/MoonshotAI/MoBA) |
| **InfLLM-V2** | [OpenBMB/infllmv2_cuda_impl](https://www.google.com/search?q=https://github.com/OpenBMB/infllmv2_cuda_impl) |

## 📝 Citation

If you find this project useful in your research, please consider citing:

```bibtex
@misc{elastic_attention_2024,
    title={Elastic Attention: ...},
    author={...},
    year={2024},
    howpublished={\url{https://anonymous.4open.science/r/Elastic-Attention-D370}}
}

```

|      |      |
| --- | --- |
|      |      |
|      |      |
|      |      |

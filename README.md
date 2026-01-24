<div align="center">

# 🚀 Elastic Attention

### Test-time Adaptive Sparsity Ratios for Efficient Transformers

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/24xx.xxxxx)
[![ModelScope](https://img.shields.io/badge/ModelScope-Collection-624aff.svg)](https://modelscope.cn/collections/LCM_group/Elastic-Attention)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-85c600.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)

</div>

---

## 📖 Introduction



**Elastic Attention** enables models to achieve both **strong performance** and **efficient inference**. 

- **High Efficiency:** Within only **12 hours** of training on $8 \times$ A800 GPUs.
- **Proven Superiority:** Verified across 3 long-context scenarios with **Llama-3.1** and **Qwen3-series** models.

## 💻 System Environment

The codebase has been strictly verified on the following high-performance computing environment:

| Component | Specification | Notes |
| :--- | :--- | :--- |
| **OS** | Ubuntu 22.04.4 LTS | Tested on ID: `ubuntu` |
| **Python** | 3.11+ | Recommended |
| **PyTorch** | 2.6.0 | Ecosystem compatible |
| **CUDA** | 12.4+ | **Required** |
| **GPU** | NVIDIA A100/H100 (80GB) | High VRAM required |

## ⚙️ Installation

### 1. Setup Python Environment

Clone the repository and set up the basic PyTorch ecosystem.

```bash
# Clone the repository
git clone [https://github.com/LCM-Lab/Elastic-Attention.git](https://github.com/LCM-Lab/Elastic-Attention.git)
cd Elastic-Attention

# Install PyTorch ecosystem (CUDA 12.4)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)

```

### 2. Install Dependencies

This project relies on `Block-Sparse-Attention` and other libraries.

> [!IMPORTANT]
> Compilation of CUDA kernels may take up to **5-10 minutes**. Please ensure `nvcc` is in your PATH.

```bash
# 2.1 Install Block-Sparse-Attention (Custom CUDA Ops)
git clone [https://github.com/mit-han-lab/Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention)
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

We provide scripts to download the Supervised Fine-Tuning (SFT) datasets via [ModelScope](https://modelscope.cn/).

### Download SFT Datasets

You can use the following Python snippets to download the processed data:

| Dataset | Code Snippet |
| --- | --- |
| **Qwen Mix SFT (64K)** | `MsDataset.load('LCM_group/qwen_mix_sft_64K6', ...)` |
| **LLaMA Mix SFT (64K)** | `MsDataset.load('LCM_group/llama_mix_sft_64K6', ...)` |

> [!TIP]
> For the training demo below, we include a small cached dataset located at:
> `elasticattn/public_data/data_cache/demo_data_qwen_packed_maxseq65536.parquet`

## 🏰 Model Zoo

Pre-trained models and checkpoints are available on ModelScope.

| Model Series | Model Scale | Link |
| --- | --- | --- |
| **Elastic-Attention Collection** | 4B / 8B | [![ModelScope](https://img.shields.io/badge/ModelScope-Collection-624aff.svg)](https://modelscope.cn/collections/LCM_group/Elastic-Attention) |

## 🏃 Training

To start training with the provided demo data, utilize the included startup script.

```bash
# Grant execution permissions
chmod +x elasticattn/run_scripts/training.sh

cd elasticattn
# Run the training script
bash run_scripts/training.sh

```

> **Configuration:** Batch size, learning rate, and other hyperparameters can be modified inside `elasticattn/run_scripts/training.sh`.

## ⚡ Quick Start (Inference)

Here is a minimal example of how to use Elastic Attention for text generation.

<details>
<summary><b>Click to expand the Inference Code</b></summary>

```python
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_sparse_model(model_path):
    config_path = f"{model_path}/config.json"
    with open(config_path, "r") as f:
        config_data = json.load(f)

    arch = config_data.get("architectures", [])
    if not arch:
        raise ValueError("No architecture found in config.json")

    arch_name = arch[0]
    print(f"Detected architecture: {arch_name}")

    if "PawLlama" in arch_name:
        from elasticattn.training.eval.modeling_flash_llama_moe import (
            PawLlamaForCausalLM, PawLlamaConfig
        )
        AutoModelForCausalLM.register(PawLlamaConfig, PawLlamaForCausalLM)
        model_cls = PawLlamaForCausalLM
        
    elif "PawQwen" in arch_name:
        from elasticattn.training.eval.modeling_flash_qwen_moe import (
            PawQwen3ForCausalLM, PawQwen3Config
        )
        AutoModelForCausalLM.register(PawQwen3Config, PawQwen3ForCausalLM)
        model_cls = PawQwen3ForCausalLM
    else:
        raise ValueError(f"Unsupported architecture: {arch_name}")

    model = model_cls.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    return model

# Usage
model_path = "****" # Replace with your checkpoint path
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = load_sparse_model(model_path)
model.eval()

# Generate
input_text = "Explain quantum mechanics in one sentence."
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

```

</details>

## ⚖️ Evaluation

We recommend using **[LOOM-Eval](https://github.com/LCM-Lab/LOOM-Eval)** for comprehensive evaluation of long-context capabilities.

```bash
# 1. Clone and Install
git clone [https://github.com/LCM-Lab/LOOM-Eval.git](https://github.com/LCM-Lab/LOOM-Eval.git)
cd LOOM-Eval
pip install -e .

# 2. Run Evaluation
loomeval.run \ 
  --model_path /path/to/model \
  --cfg_path /benchmarks/General/RULER/configs/RULER.yaml \
  --server transformers \
  --acceleration elasticattn \
  --device 0 1 2 3 4 5 6 7 \
  --gp_num 1 \
  --output_dir /path/to/results

```

## 🔗 Related Implementations

We acknowledge and reference the following open-source implementations:

| Method | Repository |
| --- | --- |
| **NSA** (Native Sparse Attention) | [XunhaoLai/native-sparse-attention-triton](https://github.com/XunhaoLai/native-sparse-attention-triton) |
| **MoBA** | [MoonshotAI/MoBA](https://github.com/MoonshotAI/MoBA) |
| **InfLLM-V2** | [OpenBMB/infllmv2_cuda_impl](https://github.com/OpenBMB/infllmv2_cuda_impl) |
| **XAttention** | [mit-han-lab/x-attention](https://github.com/mit-han-lab/x-attention) |

## 📝 Citation

If you find this project useful in your research, please consider citing:

```bibtex
@misc{elastic_attention_2026,
    title={Elastic Attention: ...},
    author={...},
    year={2026},
    howpublished={\url{[https://anonymous.4open.science/r/Elastic-Attention-D370](https://anonymous.4open.science/r/Elastic-Attention-D370)}}
}

```

# 🚀 Elastic Attention

## 🚀 Quick Start

### 💻 System Environment (Tested)
The code has been verified on the following Linux environment:
- **OS:** Ubuntu 22.04.4 LTS
- **ID:** ubuntu

### 📋 Requirements
- **Python:** 3.11+
- **PyTorch:** 2.6+ (Tested with 2.6.0)
- **CUDA:** 12.4+
- **GPU Memory:** 80GB

### ⚙️ Installation

#### 1. Setup Environment
```bash
# Clone the repository
git clone [https://anonymous.4open.science/r/Elastic-Attention-D370.git](https://anonymous.4open.science/r/Elastic-Attention-D370)
cd Elastic-Attention

# Install PyTorch ecosystem
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)

```

#### 2. Install Block-Sparse-Attention (Dependency)

```bash
# Clone the dependency
git clone [https://github.com/mit-han-lab/Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention)
cd Block-Sparse-Attention

# Install with CUDA support
# Ensure CUDA_HOME matches your local path (/usr/local/cuda-12.4/ recommended)
CUDA_HOME=/usr/local/cuda-12.4/ python setup.py install
cd ..

```

#### 3. Finalize Installation

```bash
# Install other dependencies
pip install -r requirements.txt

# Install Elastic Attention
pip install -e .

```

---

### 🏃 Training

To start training with the provided demo data, you can use the included startup script.

**Demo Data Location:**
`sparseattn/public_data/data_cache/demo_data_qwen_packed_maxseq65536.parquet`

**Run Command:**

```bash
# Grant execution permissions
chmod +x sparseattn/run_scripts/training.sh

# Run the training script
bash sparseattn/run_scripts/training.sh

```

---

### 🔗 Related Implementations & References

Below are the repositories referenced for specific implementations:

| Model / Method | Open-source Repository Link |
| --- | --- |
| **NSA** | [XunhaoLai/native-sparse-attention-triton](https://github.com/XunhaoLai/native-sparse-attention-triton) |
| **MoBA** | [MoonshotAI/MoBA](https://github.com/MoonshotAI/MoBA) |
| **InfLLM-V2** | [OpenBMB/infllmv2_cuda_impl](https://github.com/OpenBMB/infllmv2_cuda_impl) |

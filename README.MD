# NACASR: Neural Acoustic-to-Character Automatic Speech Recognition

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2023.xxxxx-b31b1b.svg)](https://arxiv.org/)

## Abstract

NACASR (Neural Acoustic-to-Character Automatic Speech Recognition) is an end-to-end neural network architecture for automatic speech recognition that directly maps acoustic features to character sequences using Connectionist Temporal Classification (CTC). The model combines Convolutional-BatchNorm-Highway-GRU (CBHG) blocks with pre-networks to achieve robust performance on low-resource languages, particularly Wolof.

**Key Contributions:**
- Novel application of CBHG architecture for direct acoustic-to-character mapping
- Efficient CTC-based training framework for sequence-to-sequence learning without attention
- Comprehensive evaluation on Wolof speech recognition dataset
- Open-source implementation with reproducible training pipeline

## Architecture Overview


<div align="center">
  <img src="https://media.licdn.com/dms/image/v2/D4E22AQHVVT5s1ABhYQ/feedshare-shrink_2048_1536/B4EZjrpp.1HgA0-/0/1756300223754?e=1760572800&v=beta&t=YHaDsjS7qrNhcUwOXJ_MaRhs_EwkgkTLt0iBU2GnduI" alt="Architecture" width="600"/>
</div>


```
Input Features (B, T, 256) 
    ↓
Pre-network (Fully Connected + ReLU + Dropout)
    ↓
CBHG Module:
├── Conv1D Banks (K=16, filters=1...16)
├── Max Pooling + Projections
├── Highway Networks (4 layers)
└── Bidirectional GRU
    ↓
Linear Projection (vocab_size)
    ↓
CTC Loss
```

## Installation

### Environment Setup

```bash
# Clone repository
git clone https://github.com/your-org/nacasr.git
cd nacasr

# Create virtual environment and install dependencies
bash setup.sh --venv --install
source venv/bin/activate
```

### Manual Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install PyTorch (choose appropriate version)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu  # CPU-only

# Install other dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Training

```bash
 
# Full training with default parameters
python run_nacasr_ctc.py --epochs 20 --batch-size 16 --lr 1e-3

 
```

## Results

### Wolof ASR Performance

| Model | WER (%) | CER (%) | Parameters | Training Time |
|-------|---------|---------|------------|---------------|
| Baseline | - | - | - | - |
| NACASR | **XX.X** | **XX.X** | XXM | XXh |

### Ablation Studies

| Component | WER (%) | Δ WER |
|-----------|---------|-------|
| Full Model | **XX.X** | - |
| w/o Highway | XX.X | +X.X |
| w/o Conv1D Banks | XX.X | +X.X |
| w/o Prenet | XX.X | +X.X |


## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Issues and Pull Requests

- 🐛 **Bug Reports**: Use the bug report template
- 💡 **Feature Requests**: Use the feature request template  
- 🔧 **Pull Requests**: Follow the PR template and ensure tests pass

## Acknowledgments

- **Dataset**: Thanks to the creators of the Wolof speech corpus
- **Architecture**: CBHG module inspired by Tacotron (Wang et al., 2017)
- **CTC Loss**: Connectionist Temporal Classification (Graves et al., 2006)
- **Funding**: [Grant/Institution acknowledgments]

## Related Work

- [Tacotron: Towards End-to-End Speech Synthesis](https://arxiv.org/abs/1703.10135)
- [Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf)
- [Deep Speech 2: End-to-End Speech Recognition](https://arxiv.org/abs/1512.02595)

## Contact

- **Author**: abdouaziz@gmail.com
- **Lab/Institution**: LAfricaMobile
- **Project Page**: [https://github.com/abdouaziz/nac_asr](https://github.com/abdouaziz/nac_asr)

---

**Keywords**: Automatic Speech Recognition, CTC, CBHG, Low-Resource Languages, Wolof, End-to-End, Neural Networks
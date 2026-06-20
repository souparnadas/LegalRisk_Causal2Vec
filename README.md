# ⚖️ Legal Risk Auditor: Causal2Vec SLM

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-F9AB00.svg)

A highly optimized Small Language Model (SLM) architecture designed to autonomously audit legal contract clauses for liability and operational risks. 

This project demonstrates advanced MLOps and model tuning techniques by fine tuning a **Mistral-7B** base model on a single free tier Google Colab T4 GPU (15GB VRAM) without encountering Out Of Memory (OOM) failures.

## 🧠 Technical Architecture

Instead of standard next token prediction or full-parameter LoRA fine tuning, this architecture utilizes a custom built **Causal2Vec Dual View Attention Head** combined with **Knowledge Distillation**.

1. **The Base Model (Frozen):** `Mistral-7B-v0.1` acts as an elite, generalized feature extractor, processing complex legal syntax into high dimensional hidden states. 
2. **Causal2Vec Attention Router:** A custom PyTorch neural classification head processes the final layer's hidden states through two distinct paths:
   * **Global View:** Captures the sentence level contextual embedding.
   * **Selective Attention View:** A trainable feed forward layer that calculates localized token importance, forcing the model to pay attention to high risk legal vocabulary.
3. **Knowledge Distillation:** The lightweight student head was trained using soft target logit alignment (Kullback–Leibler divergence) from a specialized domain teacher (`nlpaueb/legal-bert-base-uncased`), allowing the SLM to inherit structural legal nuances without the massive computational overhead.

## ⚡ Hardware Optimization & MLOps

Training a 7-Billion parameter model alongside a BERT teacher on a single 15GB VRAM GPU requires severe memory optimization. This pipeline successfully mitigated hardware bottlenecks using:
* **4-bit NF4 Quantization:** via `bitsandbytes` to load the 14.5GB Mistral model into a fraction of the memory.
* **Gradient Checkpointing:** Trading compute time for VRAM space by recalculating intermediate forward pass activations instead of storing them in memory.
* **Gradient Accumulation:** Maintaining a stable learning trajectory via an effective batch size of 32 (accumulating gradients over 32 micro-steps of batch size 1) to prevent catastrophic VRAM spikes.

## 📊 Performance Metrics

Trained on the **LEDGAR (LexGLUE)** benchmark dataset, targeting a 3-class risk profile (Low, Medium, High). 

| Epoch | Combined Loss | Test Accuracy | Macro F1-Score | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Epoch 1** | 0.8501 | 76.00% | 28.79% | Underfitting / Class Bias |
| **Epoch 2** | 0.5363 | 63.00% | 51.85% | Learning Structural Nuance |
| **Epoch 3** | **0.4782** | **85.00%** | **73.05%** | **Optimal Convergence** |

*Note: The dramatic increase in the Macro F1-Score demonstrates the architecture's success in overcoming severe dataset class imbalance and actively identifying nuanced, low-frequency risk indicators.*

## 📁 Repository Structure

* `notebooks/1_Training_and_Distillation.ipynb`: The complete end-to-end pipeline including dataset staging, custom architecture definitions, the memory-safe data collator, and the authentic training loop.
* `notebooks/2_Live_Inference.ipynb`: A clean, presentation-ready script for running real time audits on raw text.

## 🚀 Quick Start (Inference)

To run the live inference demo locally or in Colab:
1. **Install Requirements:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download the Weights:** Download `LegalAI_Brain_Epoch_3.pt` from the **Releases** tab of this repository and place it in your working directory. *(Note: The 14.5GB Mistral base model will download automatically via Hugging Face upon execution).*

3. **Run the Notebook:**
   Open `2_Live_Inference.ipynb`, instantiate the model, and pass any raw contract clause to the evaluator.

   ```python
   # Example Output
   Tested Clause: "The contractor shall indemnify and hold harmless the client against all liabilities, damages, and legal claims arising from contract breach."
   Auditor Decision: High (Legal/Liability)
   ```

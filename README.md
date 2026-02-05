# Transfer Learning for Fake News Detection: DeBERTa & LoRA

This project implements a binary classification system for fake news detection, comparing traditional machine learning baselines against a state-of-the-art DeBERTa model fine-tuned using Low-Rank Adaptation (LoRA). The goal was to demonstrate how parameter-efficient transfer learning can achieve superior performance on complex NLP tasks by leveraging diverse datasets.

## Team
* **Arda Ağaçdelen**
* **Yiğit Kaya Bağcı**

## Project Overview
We constructed a robust pipeline to classify news articles as **Real** or **Fake**. To ensure generalization, we merged three distinct datasets (ISOT, FakeNewsNet, and LIAR), creating a challenging and varied corpus. We then evaluated the performance trade-offs between classical ML approaches and modern Transformer-based methods.

## Implementation & Methodology

### Datasets
We merged and standardized three datasets to create a unified training corpus:
1.  **ISOT Fake News Dataset:** Full articles, high-context.
2.  **FakeNewsNet:** Nested structure, required custom parsing, contains specialized domain news.
3.  **LIAR:** Short statements, originally 6-class, mapped to binary (Real vs. Fake).

*   **Preprocessing:** All texts were cleaned, and we used stratified splitting to prevent data leakage between train/validation/test sets.

### Model Configuration

#### 1. Baseline Models
We optimized several traditional algorithms using **grid search** for hyperparameter tuning to establish a strong baseline:
*   **Naive Bayes**
*   **Support Vector Machines (SVM)**
*   **Logistic Regression**
*   **Random Forest**

#### 2. Transfer Learning (DeBERTa + LoRA)
We employed **DeBERTa-v3-base** as our backbone. Instead of full fine-tuning, we used **LoRA (Low-Rank Adaptation)** to efficiently adapt the model:
*   **Target Modules:** Query, Key, Value, and Output projection layers.
*   **LoRA Rank:** 8  (Efficient low-rank decomposition)
*   **LoRA Alpha:** 32 (Scaling factor)
*   **Optimization:** Hyperparameters (learning rate, batch size, weight decay) were optimized using **Optuna** trials.

## Results

Our experiments addressed the "Accuracy/Efficiency" trade-off. While baseline models performed surprisingly well on this task (~90%), the DeBERTa model with LoRA achieved the highest accuracy, demonstrating the power of contextual embeddings for detecting nuanced fake news.

| Model Category | Best Accuracy (%) |
| :--- | :--- |
| **Baseline Models** | ~90.11 - 91.26 |
| **DeBERTa + LoRA** | **92.29** |

## Environment & Tech Stack

The project was executed in **Google Colab** with the following specifications:

*   **GPU:** NVIDIA A100 Tensor Core GPU
*   **Language:** Python 3.10
*   **Key Libraries:**
    *   `transformers` & `peft`: For DeBERTa and LoRA implementation.
    *   `optuna`: For automated hyperparameter optimization.
    *   `scikit-learn`: For baseline models and metrics.
    *   `datasets`: For efficient data handling.

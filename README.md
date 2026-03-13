# Adaptive Gated Deep JSCC for Image Classification and Recovery

## 📌 Overview

This project presents an **Adaptive Gated Deep Joint Source–Channel Coding (Deep JSCC)** framework for efficient **image transmission, reconstruction, and classification** over noisy wireless communication channels.

Traditional communication systems treat **source coding (compression)** and **channel coding (error protection)** as separate processes. However, these systems often struggle when channel conditions fluctuate. Deep JSCC integrates both processes into a **single end-to-end deep learning model**, enabling more robust communication.

The proposed system introduces an **adaptive gating mechanism** that dynamically adjusts encoding strength based on the **Signal-to-Noise Ratio (SNR)** of the communication channel. This allows the system to maintain high image reconstruction quality and classification accuracy even under noisy transmission conditions.

---

# 🎯 Objectives

The main objectives of this project are:

* Develop an **Adaptive Gated Deep JSCC model** for robust wireless image transmission.
* Implement **SNR-aware dynamic noise injection** during training to simulate real-world communication channels.
* Integrate a **gating network** that dynamically adjusts encoder output based on channel conditions.
* Perform **joint optimization** using:

  * Mean Squared Error (MSE) for image reconstruction
  * Cross-Entropy Loss for classification accuracy
* Implement the complete system using **Java and Deeplearning4j** for cross-platform compatibility.

---

# 🧠 System Architecture

The proposed model consists of the following components:

### 1️⃣ Encoder Network

Extracts meaningful features from the input image and compresses them for efficient transmission.

### 2️⃣ Adaptive Gating Network

Dynamically regulates encoding strength based on the **Signal-to-Noise Ratio (SNR)** of the communication channel.

### 3️⃣ Channel Simulation Layer

Simulates a wireless communication channel using **Additive White Gaussian Noise (AWGN)**.

### 4️⃣ Decoder Network

Reconstructs the transmitted image from the encoded representation.

### 5️⃣ Classification Module

Performs semantic classification on the reconstructed image.

This architecture allows **simultaneous image reconstruction and classification while maintaining robustness against channel noise**.

---

# ⚙️ Technologies Used

### Programming Language

* Java (JDK 11+)

### Deep Learning Framework

* Deeplearning4j (DL4J)

### Libraries

* ND4J (Numerical computation)
* XChart (Data visualization)
* Java Utility Libraries

### Tools

* IntelliJ IDEA / Eclipse
* Git & GitHub

### Dataset

* MNIST Handwritten Digit Dataset

---

# 📂 Project Structure

```
Adaptive-Gated-Deep-JSCC
│
├── dataset
│   └── MNIST dataset
│
├── models
│   ├── encoder
│   ├── decoder
│   └── gating network
│
├── training
│   └── model training scripts
│
├── evaluation
│   ├── PSNR calculation
│   ├── accuracy evaluation
│   └── confusion matrix
│
├── graphs
│   ├── PSNR vs SNR
│   └── Accuracy vs SNR
│
└── README.md
```

---

# 🧪 Experimental Setup

### Training Configuration

* Optimizer: **Adam**
* Training epochs until convergence
* Dynamic **SNR-based noise injection**
* Batch training using the MNIST dataset

The model is designed to operate efficiently on CPU systems and can also leverage GPU acceleration for faster training.

---

# 📊 Performance Evaluation

The proposed **Adaptive Gated Deep JSCC model** was compared with a baseline Deep JSCC system.

### Evaluation Metrics

* Peak Signal-to-Noise Ratio (PSNR)
* Classification Accuracy
* Training Stability

### Performance Results

| Metric             | Baseline JSCC | Proposed Gated JSCC | Improvement |
| ------------------ | ------------- | ------------------- | ----------- |
| Average PSNR       | 15.79 dB      | 16.46 dB            | +0.67       |
| Average Accuracy   | 94.62%        | 96.14%              | +1.52%      |
| Training Stability | Moderate      | High                | ✓           |

The results show that the **Adaptive Gated JSCC model improves both reconstruction quality and classification accuracy**, especially in **low SNR conditions**.

---

# 📈 Visualization Results

The system generates several analysis graphs including:

* PSNR vs SNR
* Accuracy vs SNR
* Training Loss Convergence
* Confusion Matrix
* Comparative Performance Graphs

These visualizations demonstrate the robustness and efficiency of the proposed architecture across varying channel conditions.

---

# 📊 Statistical Validation

A **Friedman Non-Parametric Test** was conducted to statistically validate the performance improvement.

* Chi-Square Statistic: **6.92**
* p-value: **0.027**

Since **p < 0.05**, the performance improvement of the proposed Adaptive Gated Deep JSCC model is **statistically significant**.

---

# 💻 System Requirements

### Hardware

* Intel i5 / i7 Processor
* Minimum 4 GB RAM (8 GB recommended)
* 500 MB free storage
* Optional NVIDIA GPU for faster training

### Software

* Windows 10 / 11 (64-bit)
* Java 11 or above
* IntelliJ IDEA / Eclipse
* Deeplearning4j Framework
* Git (optional)

---

# 🚀 Applications

This system can be used in several real-world domains including:

* Wireless image communication
* Telemedicine systems
* Autonomous vehicle communication
* Remote monitoring systems
* IoT-based visual data transmission

---

# 🔮 Future Work

Possible future enhancements include:

* Training with larger datasets such as CIFAR-10 or ImageNet
* Extending the framework to **video transmission**
* Deploying the system on **edge devices and embedded platforms**
* Implementing **real-time wireless communication systems**

---

# 👨‍💻 Authors

**Henry Issac**
Mepco Schlenk Engineering College
Sivakasi, Tamil Nadu, India

Co-Authors:

* Blessa Binolin Pepsi
* Thiruguhan A

---

# 📄 License

This project is developed for **academic and research purposes**.

# 🧠 Mental Health LLM (QLoRA Fine-tuning)

This project focuses on fine-tuning a Large Language Model (Mistral-7B) for mental health and psychological support conversations using QLoRA.

---

## 🚀 Features

- Fine-tuned on real-world mental health counseling datasets
- Uses QLoRA (4-bit quantization) for efficient training
- Built with Unsloth + HuggingFace TRL
- Handles empathetic and supportive responses
- Lightweight and cost-effective training (Colab compatible)

---

## 📊 Datasets Used

- Amod/mental_health_counseling_conversations  
- nbertagnolli/counsel-chat  

These datasets contain real counseling-style conversations and therapist responses.

---

## 🏗️ Tech Stack

- Python
- HuggingFace Transformers
- PEFT (LoRA)
- Unsloth
- Datasets
- PyTorch

---

## ⚙️ Training Setup

- Base Model: Mistral-7B-Instruct (4-bit)
- Method: QLoRA
- Platform: Google Colab (T4 GPU)
- Sequence Length: 1024
- Optimizer: AdamW 8-bit

---

## 🧪 How to Run Locally

```bash
pip install -r requirements.txt
python test.py
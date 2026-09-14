# LLaMA-2 Fine-Tuning for Algorithmic Reasoning 🧠💻

An efficient fine-tuning pipeline for LLaMA-2, optimized specifically for competitive programming and algorithmic reasoning. This project leverages Quantized Low-Rank Adaptation (QLoRA) and PEFT to train the model on curated LeetCode datasets, enabling it to generate syntactically perfect, highly optimized code that passes rigorous test cases.

## 🚀 Key Features

* **Algorithmic Code Generation:** Fine-tuned to understand complex problem statements and output perfect running code (e.g., C++, Python) tailored to pass strict edge cases and hidden test cases.
* **Highly Efficient Training:** Implemented PEFT (Parameter-Efficient Fine-Tuning) to reduce computational overhead by **60%**, allowing a massive LLM to be trained on standard consumer hardware without sacrificing reasoning capabilities.
* **Robust Evaluation Pipeline:** Benchmarked against competitive programming standards, focusing on abstract syntax tree (AST) correctness, logical reasoning, and time/space complexity adherence.

## 🛠️ Tech Stack

* **Model:** LLaMA-2
* **Frameworks:** PyTorch, Hugging Face `transformers`, `peft`, `trl`
* **Techniques:** QLoRA, 4-bit Quantization (BitsAndBytes)
* **Language:** Python

## 📊 Dataset

The model was trained on a highly curated dataset of LeetCode problems encompassing:
* Dynamic Programming, Graph Theory, and Tree traversal algorithms.
* Problem descriptions paired with optimized solution templates.
* Step-by-step logical deductions ensuring the model learns the *reasoning* behind the code, not just memorizing syntax.

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Rahul-262624/Finetunning.git](https://github.com/Rahul-262624/Finetunning.git)
   cd Finetunning

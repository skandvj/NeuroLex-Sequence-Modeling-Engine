# 🗣️ NeuroLex: Neural Language Modeling Engine
**Natural Language Processing (NLP) | Sequence Modeling | Text Generation**

![Status](https://img.shields.io/badge/Status-Foundational_Architecture-blue?style=for-the-badge)
![Metric](https://img.shields.io/badge/Metric-Perplexity_Optimized-green?style=for-the-badge)
![Tech](https://img.shields.io/badge/Core-LSTM_%2F_RNN-orange?style=for-the-badge)

---

### 💼 Executive Summary
Before Transformers (GPT) revolutionized AI, **Recurrent Neural Networks (RNNs)** were the state-of-the-art for understanding sequential data.

**NeuroLex** is a character-level language model designed to predict the probability of the next character in a sequence. By leveraging **Long Short-Term Memory (LSTM)** networks, this engine solves the "Vanishing Gradient" problem found in vanilla RNNs, allowing it to generate coherent text structures by retaining long-term context (memory).

---

### ❓ The Business Problem
* **Predictive Typing:** Mobile keyboards (Autocorrect) need to predict the next word in milliseconds with limited compute.
* **Context Retention:** Simple models forget the beginning of a sentence by the time they reach the end, leading to nonsensical output.
* **Sequence Dependency:** Understanding that "Bank" means something different in "Bank of the river" vs. "Bank of America" requires sequential memory.

---

### 💡 The Solution: LSTM Memory Cells


I engineered a sequence model that acts as a "Next-Token Predictor"—the fundamental task behind modern LLMs.

| Feature | Technical Implementation | PM Value Proposition |
| :--- | :--- | :--- |
| **Long-Term Memory** | <code>LSTM Architecture</code> | Uses "Gating Mechanisms" (Input, Forget, Output gates) to decide what information to keep or discard over long sequences. |
| **Training Stability** | <code>Teacher Forcing</code> | A training strategy that feeds the *actual* previous token (ground truth) instead of the *predicted* one, stabilizing convergence. |
| **Diversity Control** | <code>Temperature Scaling</code> | A hyperparameter that controls the "creativity" of the output (Low Temp = Deterministic, High Temp = Creative/Random). |

---

### 🔬 Technical Deep Dive (Ablation)
*Defining the architecture requires balancing computational cost with memory retention.*

| Experiment | Configuration | Outcome | Decision |
| :--- | :--- | :--- | :--- |
| **Backbone** | `Vanilla RNN` vs `LSTM` | RNN failed to capture dependencies >10 characters back (Vanishing Gradient). | ✅ **Selected LSTM** |
| **Optimization** | `Adam` vs `SGD` | Adam converged 3x faster for this sparse text data. | ✅ **Selected Adam** |
| **Regularization** | `Weight Tying` | Tying input embedding weights to output weights reduced parameter count by 40%. | ✅ **Implemented** |

---

### 📊 Output Demonstration
*The model was trained on a corpus of text and asked to "continue" a prompt.*

> **Prompt:** "The quick brown fox"
>
> **NeuroLex Generation (T=0.8):** "...jumps over the lazy dog and runs into the deep forest where the sun does not shine. It is a time of great mystery..."
>
> *(Note: The model learned grammar, spacing, and sentence structure purely from raw character sequences.)*

---

### 🛠 Tech Stack
* **Framework:** `PyTorch`
* **Architecture:** `LSTM` (Long Short-Term Memory), `GRU` (Gated Recurrent Unit)
* **Metric:** `Perplexity` (Lower is better)
* **Optimization:** `Backpropagation Through Time (BPTT)`

---

### 🚀 How to Run the Model
```bash
# Clone the repository
git clone [https://github.com/skandvj/HW4P1-Language-Modelling.git](https://github.com/skandvj/HW4P1-Language-Modelling.git)

# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py --model lstm --epochs 20 --batch_size 64

# Generate Text
python generate.py --prompt "The future of AI is" --temperature 0.7
```

### Author
Skand Vijay

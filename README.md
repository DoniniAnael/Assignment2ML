# 🧠 Phishing LLM Evaluation & Chatbot System

A full pipeline for **benchmarking**, **fine-tuning**, and **interacting** with open-source LLMs (e.g. LLaMA 3) to explore misuse potential like phishing message generation. Includes GUI chatbot, Arena mode, refusal benchmarking, and LoRA-based fine-tuning.

---

## 📦 Key Components

### 🔬 1. Benchmarks
- **ARC-Challenge evaluation**: Reasoning test to track model performance.
- **Refusal benchmark**: Detects if model refuses or complies with harmful prompts.
- **Phishing quality scoring**: Uses GPT-based grader or Arena user votes.
- **Arena Benchmark**: Uses Arena user votes.

### 🧪 2. Fine-Tuning Pipeline
- LoRA fine-tuning on `meta-llama/Llama-3.2-1B-Instruct`
- Merging adapters and generation/detection tests
- Fully local HuggingFace-based training and inference (see `Llama_Ft_instruct_v2.py`)
- Merged model exported to GGUF format for use with llama.cpp and Ollama
-⚠️ GGUF model (llama3-1b-spamgen.gguf) is not included in this repository

### 🤖 3. Interactive Chatbot
- PDF/image input with OCR
- Voice input + TTS output
- RAG context retrieval (FAISS + SentenceTransformer)
- Real-time DuckDuckGo search
- Toggleable Arena mode for A/B testing

---

## 🔧 Setup

### 1. Clone Repo

```bash
git clone https://github.com/your-username/phishing-llm-eval.git
cd phishing-llm-eval


# Install Ollama
https://ollama.com

# Pull models
ollama pull llama3.2:1b
ollama pull llama3.2:3b
ollama pull llama3-1b-spamgen

#Setup Tesseract (OCR)
Windows: Download and install. https://github.com/tesseract-ocr/tesseract 

Add the install path to your PATH environment variable.

Default: "C:\Program Files\Tesseract-OCR\tesseract.exe"

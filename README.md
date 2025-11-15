# AmbedkarGPT - AI Intern Assignment (Kalpit Pvt Ltd)

### 👨‍💻 Author: Anchal devi

## Overview
This is a **Retrieval-Augmented Generation (RAG)** system built with **LangChain**, **ChromaDB**, **HuggingFace embeddings**, and **Ollama (Mistral 7B)**.

It reads a **speech by Dr. B.R. Ambedkar**, splits it into chunks, creates embeddings, stores them locally, and uses them to answer user questions — all **100% offline**.

---

## 🧠 Tech Stack
- Python 3.8+
- LangChain
- ChromaDB (local vector store)
- HuggingFace Sentence Transformers
- Ollama + Mistral 7B (local LLM)

---

✅ Setup Instructions — How to Clone & Run the Project

Anyone who wants to run your RAG chatbot can follow these steps.

📥 1. Clone the Repository

Open a terminal and run:

git clone https://github.com/Anchalde/AmbedkarGPT.git
cd AmbedkarGPT-RAG

🐍 2. Create a Virtual Environment
python -m venv venv


Activate it:

🔹 Windows:
venv\Scripts\activate

🔹 macOS / Linux:
source venv/bin/activate

📦 3. Install Dependencies

Make sure you're inside the activated venv, then run:

pip install -r requirements.txt

🤖 4. Install Ollama
🔹 Windows:

Download installer → https://ollama.com/download

🔹 macOS / Linux:
curl -fsSL https://ollama.ai/install.sh | sh

🧠 5. Pull the Mistral Model:

ollama pull mistral


This downloads the LLM used by your project.

📂 6. Make Sure “speech.txt” Exists

Ensure speech.txt (Ambedkar's document) is inside the project folder.

▶️ 7. Run the Project:
python main.py


You should see:

Setup complete. You can now ask questions!


Then type any question:

What does Ambedkar say about caste?

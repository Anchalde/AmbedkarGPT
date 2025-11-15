"""
AmbedkarGPT - AI Intern Assignment (Kalpit Pvt Ltd)
Author: Nick (Maneesh Reddy Alugupalli)

A clean, warning-free LangChain RAG prototype using:
 - HuggingFace embeddings (sentence-transformers/all-MiniLM-L6-v2)
 - ChromaDB (local vector store)
 - Ollama (Mistral 7B LLM)
"""

import os
import warnings
import logging

# ──────────────────────────────
# Silence unwanted warnings/logs
# ──────────────────────────────
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "false"
logging.getLogger("chromadb.telemetry").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)

# ──────────────────────────────
# LangChain Imports
# ──────────────────────────────
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA


def build_vector_store(file_path: str, persist_directory: str = "db"):
    """Load text, split into chunks, create embeddings, and store in ChromaDB."""
    print("\n📘 Loading document...")
    loader = TextLoader(file_path)
    documents = loader.load()

    print("✂️ Splitting document into chunks...")
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=300,
        chunk_overlap=50,
        length_function=len
    )
    docs = splitter.split_documents(documents)

    print("🧠 Creating embeddings using HuggingFace...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("💾 Storing vectors locally in ChromaDB...")
    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)

    return vectorstore


def create_qa_chain(vectorstore):
    """Create RetrievalQA chain using Ollama’s Mistral 7B."""
    print("\n⚙️ Initializing Ollama (Mistral 7B)...")
    llm = Ollama(model="mistral")

    print("🔍 Setting up RetrievalQA pipeline...")
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True
    )
    return qa


def main():
    persist_dir = "db"
    file_path = "speech.txt"

    # Load or build vector store
    if not os.path.exists(persist_dir):
        vectorstore = build_vector_store(file_path, persist_dir)
    else:
        print("\n📦 Loading existing ChromaDB store...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    qa_chain = create_qa_chain(vectorstore)

    print("\n✅ Setup complete. You can now ask questions!")
    print("---------------------------------------------------")

    while True:
        query = input("\n❓ Ask a question (or type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            print("👋 Exiting. Goodbye!")
            break

        # Use invoke() instead of deprecated __call__()
        result = qa_chain.invoke({"query": query})

        print("\n💬 Answer:")
        print(result["result"])

        print("\n📄 Source context:")
        for doc in result["source_documents"]:
            print("-", doc.page_content.strip()[:120], "...\n")


if __name__ == "__main__":
    main()

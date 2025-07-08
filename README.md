# Chat with Multiple PDFs using LangChain and LLaMA3 (Groq API)  

This project demonstrates how to build a chatbot that can interact with multiple PDF documents using the LLaMA3 model via the Groq API, powered by LangChain. It supports uploading, indexing, and querying PDFs through a conversational interface built with Streamlit.

## 🔍 Features

- Upload and interact with **multiple PDF files**
- Convert PDF content into **vector embeddings** for semantic search
- Query documents using **natural language**
- Uses **LLaMA3-8B** model from **Groq**
- Clean and interactive **Streamlit UI**

## 🚀 Tech Stack

- **Python**
- **LangChain**
- **Groq LLaMA3 API**
- **FAISS** for vector storage
- **Streamlit** for frontend
- **PyPDF2** for reading PDF content

## 📦 Installation

```bash
git clone https://github.com/yourusername/pdf-chatbot-llama3.git
cd pdf-chatbot-llama3
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
pip install -r requirements.txt

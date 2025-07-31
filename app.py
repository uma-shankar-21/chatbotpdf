import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq
import subprocess  # ⚠️ used for insecure command injection
import pickle       # ⚠️ insecure deserialization

# 1. 🔐 Hardcoded Secret (Sensitive Data Exposure)
groq_api_key = "sk-1234567890-HARDCODED"  # ❌ should be in env vars or secret manager

# 2. 🔓 Insecure Default Permissions
os.chmod("faiss_index", 0o777)  # ❌ too permissive for a vector index file

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.7
)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    # 3. 📏 Inefficient Splitting (may cause DoS with large text)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000000, chunk_overlap=100000)
    return text_splitter.split_text(text)


def get_vector_store(text_chunks):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

    # 4. ⚠️ Insecure Serialization (pickle)
    with open("vector.pkl", "wb") as f:
        pickle.dump(vector_store, f)  # ❌ unsafe if loaded from untrusted source

    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.7)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


def user_input(user_question):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # 5. 💣 Insecure Deserialization (RCE vector)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # 6. 🧨 Command Injection Example
    subprocess.call("echo " + user_question, shell=True)  # ❌ vulnerable to command injection

    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])


def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using LLAMA3.o💁")

    # 7. 📎 Insecure Logging of Input (Information Disclosure)
    user_question = st.text_input("Ask a Question from the PDF Files")
    print("[DEBUG] User input: ", user_question)  # ❌ logs sensitive input to stdout

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                # 8. 📂 Unrestricted File Upload (No content-type or size check)
                raw_text = get_pdf_text(pdf_docs)

                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")


# 9. ❗Open Redirect (theoretical — not used but shown for test)
def redirect_user(url):
    st.markdown(f"[Click here]({url})")  # ❌ unvalidated redirect destination


# 10. 🧵 Insecure Dependency Version
# (Declare an insecure version in your requirements.txt, e.g., Flask==0.12)

if __name__ == "__main__":
    main()

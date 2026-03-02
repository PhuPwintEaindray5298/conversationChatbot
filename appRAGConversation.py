import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from langchain_classic.chains import (
    create_retrieval_chain,
    create_history_aware_retriever
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


# --------------------------------------------------
# ENV SETUP
# --------------------------------------------------
load_dotenv()

os.environ["HUGGINGFACE_API_KEY"] = os.getenv("HUGGINGFACE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------
st.set_page_config(page_title="Conversational RAG with PDF", layout="wide")

st.title("📄 Conversational RAG with PDF Upload")
st.write("Upload a PDF and chat with its contents using memory-aware RAG.")

session_id = st.text_input("Session ID", value="default_session")

if "store" not in st.session_state:
    st.session_state.store = {}

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type="pdf",
    accept_multiple_files=False
)

# --------------------------------------------------
# MODELS
# --------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile"
)

# --------------------------------------------------
# CHAT HISTORY HANDLER
# --------------------------------------------------
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# --------------------------------------------------
# PDF PROCESSING & RAG SETUP
# --------------------------------------------------
if uploaded_file:

    # Save uploaded PDF safely
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        pdf_path = tmp_file.name

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    st.success(f"Loaded {len(documents)} pages from PDF")

    # Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=500
    )
    splits = splitter.split_documents(documents)

    # Vector store
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )
    retriever = vectorstore.as_retriever()

    # --------------------------------------------------
    # HISTORY-AWARE RETRIEVER
    # --------------------------------------------------
    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Given a chat history and the latest user question, "
                "rewrite the question so it can be understood without "
                "the chat history. Do not answer the question."
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_prompt
    )

    # --------------------------------------------------
    # QA PROMPT
    # --------------------------------------------------
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a question-answering assistant. "
                "Use the provided context to answer the question. "
                "If the answer is unknown, say you don't know. "
                "Use at most three sentences.\n\n{context}"
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        qa_chain
    )

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    # --------------------------------------------------
    # CHAT UI
    # --------------------------------------------------
    user_input = st.text_input("Ask a question about the PDF")

    if user_input:
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={
                "configurable": {"session_id": session_id}
            }
        )

        st.markdown("### ✅ Answer")
        st.write(response["answer"])

        st.markdown("### 💬 Chat History")
        for msg in get_session_history(session_id).messages:
            st.write(f"**{msg.type.capitalize()}**: {msg.content}")
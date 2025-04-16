import os
import shutil
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import MessagesPlaceholder

# Initialize session states
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def save_uploaded_document(uploaded_file):
    dir_name = "Transcripts"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    try:
        file_path = os.path.join(dir_name, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving document: {e}")
        return None

def load_docs(directory):
    documents = []
    
    # Supported file types and their loaders
    loaders = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        '.md': UnstructuredMarkdownLoader
    }

    for root, _, files in os.walk(directory):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in loaders:
                try:
                    loader = loaders[ext](os.path.join(root, file))
                    documents.extend(loader.load())
                except Exception as e:
                    st.error(f"Error loading {file}: {e}")
    
    return documents

def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

def chroma_db_store(load_db=False):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    CHROMA_PATH = "chroma_db"

    if os.path.exists(CHROMA_PATH):
        load_db = True

    if load_db:
        vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        return vectordb
    else:
        dir = 'Transcripts'
        documents = load_docs(dir)
        if not documents:
            raise ValueError("No valid documents found in Transcripts directory")
        docs = split_docs(documents)
        vectordb = Chroma.from_documents(
            documents=docs, embedding=embeddings, persist_directory=CHROMA_PATH
        )
        vectordb.persist()
        return vectordb

def load_text_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash", convert_system_message_to_human=True)

def get_result(vectordb, query, chat_history=[], explain_to_kid=False):
    text_llm = load_text_llm()
    retriever = vectordb.as_retriever()

    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    contextualize_q_chain = contextualize_q_prompt | text_llm | StrOutputParser()

    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    {context}""" + (" Explain in simple terms suitable for a child." if explain_to_kid else "")

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    def contextualized_question(input: dict):
        if input.get("chat_history"):
            return contextualize_q_chain
        else:
            return input["question"]

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        RunnablePassthrough.assign(
            context=contextualized_question | retriever | format_docs
        )
        | qa_prompt
        | text_llm
    )

    res = rag_chain.invoke({"question": query, "chat_history": chat_history})
    chat_history.extend([HumanMessage(content=query), res])
    
    return res.content, chat_history

def get_llm_response(query, api_key, explain_to_kid=False):
    os.environ["GOOGLE_API_KEY"] = api_key
    vectordb = chroma_db_store()
    result, updated_chat_history = get_result(
        vectordb, query, 
        st.session_state.chat_history, explain_to_kid
    )
    st.session_state.chat_history = updated_chat_history
    return result

# Streamlit UI
st.set_page_config(page_title="Datadoc", page_icon=":robot:", layout="wide")

# Sidebar
st.sidebar.title("Options")
uploaded_docs = st.sidebar.file_uploader(
    "Upload Documents",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True
)
if uploaded_docs:
    for doc in uploaded_docs:
        save_uploaded_document(doc)
    st.sidebar.success(f"Uploaded {len(uploaded_docs)} documents")

process_btn = st.sidebar.button("Process Documents")
if process_btn:
    if os.path.exists("chroma_db"):
        shutil.rmtree("chroma_db")
    try:
        chroma_db_store()
        st.sidebar.success("Documents processed successfully!")
    except Exception as e:
        st.sidebar.error(f"Processing failed: {str(e)}")

clear_btn = st.sidebar.button("Clear Conversation")

# Main interface
st.title("Datadoc: Your AI DOC Assistant")

# Chat handling
if clear_btn:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state.chat_history = []

container = st.container()
response_container = st.container()

with container:
    with st.form(key='chat_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        explain_kid = st.toggle("Child Mode", key='kid_toggle')
        submit_btn = st.form_submit_button("Send")
        api_key = st.text_input("Enter your Google API key:", type="password")

        if submit_btn:
            if not api_key:
                st.error("Please enter API key")
            else:
                response = get_llm_response(user_input, api_key, explain_kid)
                st.session_state.past.append(user_input)
                st.session_state.generated.append(response)

with response_container:
    for i in range(len(st.session_state['generated'])):
        st.markdown(f"**You:** {st.session_state['past'][i]}")
        with st.container(border=True):
            st.markdown(f"{st.session_state['generated'][i]}")
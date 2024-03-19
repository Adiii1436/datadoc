import os
import uuid
import shutil
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

def save_string_to_txt(input_string, directory="Transcripts"):
    unique_filename = str(uuid.uuid4()) + ".txt"
    filepath = os.path.join(directory, unique_filename)

    try:
        with open(filepath, "w") as file:
            file.write(input_string)
        print(f"String successfully saved to {filepath}")
    except Exception as e:
        print(f"Error: {e}")

def load_docs(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()
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
        docs = split_docs(documents)
        vectordb = Chroma.from_documents(
            documents=docs, embedding=embeddings, persist_directory=CHROMA_PATH
        )
        vectordb.persist()

        return vectordb
    


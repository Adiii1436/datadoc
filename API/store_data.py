from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents

def split_docs(documents,chunk_size=1000,chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

def chroma_db_store(docs):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    persist_directory = "chroma_db"

    vectordb = Chroma.from_documents(
        documents=docs, embedding=embeddings, persist_directory=persist_directory
    )

    vectordb.persist()

    return vectordb

def store_data(dir,image=False):
    documents = load_docs(dir)

    docs = split_docs(documents)

    vectordb = chroma_db_store(docs)

    return vectordb

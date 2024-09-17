
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

CHROMA_PERSIST_DIR = "chroma_db"

def get_retriever():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.environ["API_KEY"])
    
    vectorstore = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
    
    if vectorstore._collection.count() == 0:
        load_documents("Dataset")
    
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

def load_documents(directory):
    loader = DirectoryLoader(directory, glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200, add_start_index=True)
    texts = text_splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.environ["API_KEY"])
    vectorstore = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
    vectorstore.add_documents(texts)
    vectorstore.persist()

def update_document_set(new_directory):
    
    loader = DirectoryLoader(new_directory, glob="./*.pdf", loader_cls=PyPDFLoader)
    new_documents = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200, add_start_index=True)
    new_texts = text_splitter.split_documents(new_documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.environ["API_KEY"])
    vectorstore = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
    vectorstore.add_documents(new_texts)
    vectorstore.persist()

    print(f"Added {len(new_texts)} new document chunks to the database.")

if __name__ == "__main__":
    update_document_set("Dataset")
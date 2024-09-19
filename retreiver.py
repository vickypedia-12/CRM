
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, UnstructuredMarkdownLoader, 
    UnstructuredExcelLoader, UnstructuredPowerPointLoader, UnstructuredCSVLoader, UnstructuredEPubLoader
    )
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
    loaders = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".md": UnstructuredMarkdownLoader,
        ".xlsx": UnstructuredExcelLoader,
        ".pptx": UnstructuredPowerPointLoader,
        ".csv": UnstructuredCSVLoader,
        ".epub": UnstructuredEPubLoader,
    }

    documents = []
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        file_extension = os.path.splitext(file)[1].lower()
        if file_extension in loaders:
            loader = loaders[file_extension](file_path)
            documents.extend(loader.load_and_split())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200, add_start_index=True)
    texts = text_splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.environ["API_KEY"])
    vectorstore = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
    vectorstore.add_documents(texts)

def update_document_set(new_directory):
    
    loaders = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".md": UnstructuredMarkdownLoader,
        ".xlsx": UnstructuredExcelLoader,
        ".pptx": UnstructuredPowerPointLoader,
        ".csv": UnstructuredCSVLoader,
        ".epub": UnstructuredEPubLoader,
    }
    
    new_documents = []
    for file in os.listdir(new_directory):
        file_path = os.path.join(new_directory, file)
        file_extension = os.path.splitext(file)[1].lower()
        if file_extension in loaders:
            loader = loaders[file_extension](file_path)
            new_documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200, add_start_index=True)
    new_texts = text_splitter.split_documents(new_documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.environ["API_KEY"])
    vectorstore = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
    vectorstore.add_documents(new_texts)

    print(f"Added {len(new_texts)} new document chunks to the database.")

if __name__ == "__main__":
    update_document_set("DB1")
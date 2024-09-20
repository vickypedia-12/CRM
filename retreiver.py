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

class CustomerRAG:
    def __init__(self, customer_id):
        self.customer_id = customer_id
        self.chroma_persist_dir = f"chroma_db_customer{customer_id}"
        self.dataset_dir = f"Dataset_customer{customer_id}"
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.environ["API_KEY"])
        self.vectorstore = None

    def get_retriever(self):
        if not self.vectorstore:
            self.vectorstore = Chroma(persist_directory=self.chroma_persist_dir, embedding_function=self.embeddings)
            
            if self.vectorstore._collection.count() == 0:
                self.load_documents()
        
        return self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    def load_documents(self):
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
        for file in os.listdir(self.dataset_dir):
            file_path = os.path.join(self.dataset_dir, file)
            file_extension = os.path.splitext(file)[1].lower()
            if file_extension in loaders:
                loader = loaders[file_extension](file_path)
                documents.extend(loader.load_and_split())

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200, add_start_index=True)
        texts = text_splitter.split_documents(documents)

        if not self.vectorstore:
            self.vectorstore = Chroma(persist_directory=self.chroma_persist_dir, embedding_function=self.embeddings)
        
        self.vectorstore.add_documents(texts)

    def update_document_set(self, new_directory):
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

        if not self.vectorstore:
            self.vectorstore = Chroma(persist_directory=self.chroma_persist_dir, embedding_function=self.embeddings)
        
        self.vectorstore.add_documents(new_texts)
        print(f"Added {len(new_texts)} new document chunks to the database for customer {self.customer_id}.")

class RAGChatbotManager:
    def __init__(self):
        self.customer_rags = {}

    def get_customer_rag(self, customer_id):
        if customer_id not in self.customer_rags:
            self.customer_rags[customer_id] = CustomerRAG(customer_id)
        return self.customer_rags[customer_id]

    def update_customer_dataset(self, customer_id, new_directory):
        customer_rag = self.get_customer_rag(customer_id)
        customer_rag.update_document_set(new_directory)


rag_manager = RAGChatbotManager()

def get_retriever(customer_id):
    customer_rag = rag_manager.get_customer_rag(customer_id)
    return customer_rag.get_retriever()

if __name__ == "__main__":
    # Example usage
    rag_manager.update_customer_dataset("customer1", "Dataset_customer1")
    rag_manager.update_customer_dataset("customer2", "Dataset_customer2")
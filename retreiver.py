
import time
from typing import List
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, UnstructuredMarkdownLoader, 
    UnstructuredExcelLoader, UnstructuredPowerPointLoader, UnstructuredCSVLoader, UnstructuredEPubLoader
)
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import random
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

class RateLimiter:
    def __init__(self, max_requests_per_minute, max_tokens_per_minute):
        self.max_requests_per_minute = max_requests_per_minute
        self.max_tokens_per_minute = max_tokens_per_minute
        self.request_timestamps = []
        self.token_count = 0
        self.last_reset = time.time()

    def wait(self, tokens):
        current_time = time.time()
        

        if current_time - self.last_reset >= 60:
            self.request_timestamps = []
            self.token_count = 0
            self.last_reset = current_time
        

        self.request_timestamps = [ts for ts in self.request_timestamps if current_time - ts < 60]
        

        while len(self.request_timestamps) >= self.max_requests_per_minute:
            time.sleep(1)
            current_time = time.time()
            self.request_timestamps = [ts for ts in self.request_timestamps if current_time - ts < 60]
        

        while self.token_count + tokens > self.max_tokens_per_minute:
            time.sleep(1)
            current_time = time.time()
            if current_time - self.last_reset >= 60:
                self.token_count = 0
                self.last_reset = current_time
        
        self.request_timestamps.append(current_time)
        self.token_count += tokens

class CustomerRAG:
    def __init__(self, customer_id):
        self.customer_id = customer_id
        self.chroma_persist_dir = f"chroma_db_customer{customer_id}"
        self.dataset_dir = f"Dataset_customer{customer_id}"
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.environ["GOOGLE_API_KEY2"])
        self.vectorstore = None
        self.batch_size = 7
        self.delay = 45
        self.rate_limiter = RateLimiter(max_requests_per_minute=10, max_tokens_per_minute=10000)

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

        labeled_texts = self.label_documents(texts)
        if not self.vectorstore:
            self.vectorstore = Chroma(persist_directory=self.chroma_persist_dir, embedding_function=self.embeddings)
        
        self.vectorstore.add_documents(labeled_texts)

    def label_documents(self, documents: List[Document]) -> List[Document]:
        print(f"Labeling {len(documents)} documents for customer {self.customer_id}...")
        labeling_prompt = PromptTemplate(
            input_variables=["text"],
            template="Given the following text, provide a short label (1-2 words) that best describes its main topic or content:\n\n{text}\n\nLabel:"
        )
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=os.environ["GOOGLE_API_KEY2"], temperature=0.2)
        labeling_chain = labeling_prompt | model | StrOutputParser()

        labeled_documents = []
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i+self.batch_size]
            print(f"Processing batch {i//self.batch_size + 1} of {len(documents)//self.batch_size + 1}")
            
            for doc in batch:
                retries = 5
                base_wait_time = 1
                while retries > 0:
                    try:
                        estimated_tokens = len(doc.page_content.split()) + 50
                        self.rate_limiter.wait(estimated_tokens)
                        
                        label = labeling_chain.invoke(doc.page_content).strip()
                        doc.metadata["label"] = label
                        labeled_documents.append(doc)
                        break
                    except Exception as e:
                        print(f"Error labeling document: {e}")
                        retries -= 1
                        if retries > 0:
                            wait_time = base_wait_time * (2 ** (5 - retries)) + random.uniform(0, 1)
                            print(f"Retrying in {wait_time:.2f} seconds...")
                            time.sleep(wait_time)
                        else:
                            print(f"Failed to label document after 5 attempts. Skipping.")
                            doc.metadata["label"] = "Unlabeled"
                            labeled_documents.append(doc)
                    else:
                        #guess who's back
                        print(f"Successfully labeled document: {label}")

            if i + self.batch_size < len(documents):
                print(f"Waiting for {self.delay} seconds before processing next batch...")
                time.sleep(self.delay)
        
        return labeled_documents
    

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

        labeled_texts = self.label_documents(new_texts)
        if not self.vectorstore:
            self.vectorstore = Chroma(persist_directory=self.chroma_persist_dir, embedding_function=self.embeddings)
        
        self.vectorstore.add_documents(labeled_texts)
        print(f"Added {len(labeled_texts)} new document chunks to the database for customer {self.customer_id}.")

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
    rag_manager.update_customer_dataset("1", "Dataset_customer1")
    rag_manager.update_customer_dataset("2", "Dataset_customer2")
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

def get_retriever():
   
    loader = DirectoryLoader('Dataset', glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200, add_start_index=True)
    texts = text_splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.environ["GOOGLE_API_KEY"])
    vectorstore = Chroma.from_documents(texts, embeddings)

    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

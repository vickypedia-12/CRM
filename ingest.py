from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

loader = DirectoryLoader('Dataset', glob="./*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200, add_start_index=True)
texts = text_splitter.split_documents(documents)

print(len(texts[0].page_content))
embedings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1",model_kwargs={"trust_remote_code":True})
vectorstore = Chroma.from_documents(documents = texts, embedding =  embedings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

def get_retrieved_docs(query):
    retrieved_docs = retriever.invoke(query)
    return retrieved_docs

query = 'What is Thakur College of Engineering and Technologys mission and vision?'

get_retrieved_docs(query)

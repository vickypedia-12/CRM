from langchain_chroma import Chroma
import os
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_vertexai import ChatVertexAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
import pandas as pd

load_dotenv()
gemini_api_key = genai.configure(api_key=os.environ["API_KEY"])

loader = DirectoryLoader('Dataset', glob="./*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200, add_start_index=True)
texts = text_splitter.split_documents(documents)

print(len(texts[0].page_content))
embedings = GooglePalmEmbeddings()
vectorstore = Chroma.from_documents(documents = texts, embedding = embedings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

prompt_template = """
    <s>[INST]f"You are a helpful and informative chatbot that answers questions using text from the reference passage included below. "
    f"Respond in a complete sentence and make sure that your response is easy to understand for everyone. "
    f"Maintain a friendly and conversational tone. If the passage is irrelevant, feel free to ignore it.\n\n"
    f"PASSAGE: '{context}'\n"
    f"QUESTION: '{query}'\n"
    f"ANSWER:"'
    </s>[INST]
"""

prompt = PromptTemplate(template=prompt_template,
     input_variables=['context', 'question'])

llm = ChatVertexAI(model="gemini-1.5-flash", project_id= "crm-rag")

# def format_docs(retriever):
#     return "\n\n".join(doc.page_content for doc in retriever) 

rag_chain = (

    {"context": retriever | retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

result = rag_chain.invoke(query="What is Thakur College of Engineering and Technologys mission and vision?")
print(result)


# def get_retrieved_docs(query):
#     retrieved_docs = retriever.invoke(query)
#     return [doc.page_content for doc in retrieved_docs]




# def make_rag_prompt(query, retrieved_passages):
#     retrieved_passage = ' '.join(retrieved_passages),
#     prompt = (

#     )
#     return prompt

# def generate_response(user_prompt):
#     model = genai.GenerativeModel('gemini-flash-1.5')
#     answer = model.generate_content(user_prompt)
#     return answer.text

# def get_response(query):
#     retrieved_passages = get_retrieved_docs(query)
#     user_prompt = make_rag_prompt(query, retrieved_passages)
#     response = generate_response(user_prompt)
#     return response

# def generate_answer(query):
#     relevant_text = get_retrieved_docs(query)
#     text = " ".join(relevant_text)
#     prompt = make_rag_prompt(query, retrieved_passages=text)
#     answer = generate_response(prompt)
#     return answer

# answer = generate_answer(query = "What is Thakur College of Engineering and Technologys mission and vision?")
# print(answer)
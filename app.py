from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from .retreiver import get_retriever
import os


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define the model
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=os.environ["API_KEY"], temperature=0.2)

# Define the prompt template
template = """
<s>[INST]
"You are a helpful and informative chatbot that answers questions using text from the reference passage included below. "
"Respond in a complete sentence and make sure that your response is easy to understand for everyone, elaborate more from your side. "
"Maintain a friendly and conversational tone. If the passage is irrelevant, feel free to ignore it.\n\n"
"PASSAGE: '{context}'\n"
"QUESTION: '{query}'\n"
"ANSWER:" 
</s>[INST]
"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Create the RAG chain
def generate_response(query):
    retriever = get_retriever()
    rag_chain = (
        {
            "context": retriever | format_docs,
            "query": RunnablePassthrough(),
        }
        | QA_CHAIN_PROMPT
        | model
        | StrOutputParser()
    )
    result = rag_chain.invoke(query)
    return result

# Example usage
query = "Give me details about head of research and development"
response = generate_response(query)
print(response)

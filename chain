from langchain import SerpAPIWrapper
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import StringPromptTemplate
from langchain import LLMChain, PromptTemplate
from langchain.chains import RetrievalQA
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.retrievers import WikipediaRetriever
from langchain import LLMMathChain
import weaviate
import json


WEAVIATE_URL= "https://propertyqna-mym99l34.weaviate.network"

client = weaviate.Client(
    url = WEAVIATE_URL,
    auth_client_secret=weaviate.AuthApiKey(api_key="KGy1GFBgfu0hPOHKgreZysSV06hHF0FK4fw2"),  # Replace w/ your Weaviate instance API key
    additional_headers = {
        "X-HuggingFace-Api-Key": "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"  # Replace with your inference API key
    }
)

#general qna chain
qna_template = """This is a conversation between a human and a bot:

{chat_history}

Answer all questions related to buying a house in Malaysia, {input}:
"""

qna_prompt = PromptTemplate(input_variables=["input", "chat_history"], template=qna_template)
memory = ConversationBufferMemory(memory_key="chat_history")
chain_type_kwargs1 = {"prompt": qna_prompt}
vectorstore = Weaviate(client,"Qna", "answer", "question")
qna_qachain = RetrievalQA.from_chain_type(llm,chain_type="stuff", chain_type_kwargs=chain_type_kwargs1, retriever=vectorstore.as_retriever(), memory= memory)

#general wikipedia chain
malaysia_retriever = WikipediaRetriever()
malaysia_template = """This is a conversation between a human and a bot:

{chat_history}

Answer all questions related to Malaysia, for example history, population, etc {input}:
"""
malaysia_prompt = PromptTemplate(input_variables=["input", "chat_history"], template=malaysia_template)
memory = ConversationBufferMemory(memory_key="chat_history")
chain_type_kwargs2 = {"prompt": malaysia_prompt}
malaysia_qachain = RetrievalQA.from_chain_type(llm,chain_type_kwargs=chain_type_kwargs2, chain_type="stuff", retriever=malaysia_retriever, memory=memory)

#general places chain
places_retriever = GoogleSerperAPIWrapper(type="places")
places_template = """This is a conversation between a human and a bot:

{chat_history}

Answer all questions related to places queries in Malaysia, for instant nearby malls, nearby public transportations, etc {input}:
"""

places_prompt = PromptTemplate(input_variables=["input", "chat_history"], template=places_template)
memory = ConversationBufferMemory(memory_key="chat_history")
chain_type_kwargs3 = {"prompt": places_prompt}

places_qachain = RetrievalQA.from_chain_type(llm,chain_type_kwargs=chain_type_kwargs3, chain_type="stuff", retriever=places_retriever, memory=memory)

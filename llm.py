import transformers
import torch
import weaviate
import json
import re
import sys
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate
from langchain import SerpAPIWrapper
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from langchain.agents import BaseMultiActionAgent
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.prompts import StringPromptTemplate
from langchain import LLMChain, PromptTemplate
from langchain.vectorstores.weaviate import Weaviate
from langchain.vectorstores import Weaviate
from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.tools import WikipediaQueryRun
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.utilities import WikipediaAPIWrapper
from langchain import LLMMathChain
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import AutoTokenizer
from torch import cuda, bfloat16
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline

name = 'mosaicml/mpt-7b-chat'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=True)
# config.attn_config['attn_impl'] = 'triton'
config.init_device = 'cuda:0' # For fast initialization directly on GPU!
# config.init_device = 'meta' # For fast initialization directly on GPU!

model = transformers.AutoModelForCausalLM.from_pretrained(
    name,
    config=config,
    trust_remote_code=True,
    torch_dtype=bfloat16,
    # model_max_length =4096
)
model.eval()
model.to(device)
print(f"Model loaded on {device}")
    
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    # mtp-7b is trained to add "<|endoftext|>" at the end of generations
stop_token_ids = tokenizer.convert_tokens_to_ids(["<|endoftext|>"])
    
    # define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False
    
stopping_criteria = StoppingCriteriaList([StopOnTokens()])  
generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    device=device,
        
    stopping_criteria=stopping_criteria,  # without this model will ramble
    temperature=1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    top_p=0.30,  # select from top tokens whose probability add up to 15%
    top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
    max_new_tokens=1024,  # mex number of tokens to generate in the output
    repetition_penalty=1  # without this output begins repeating
)

llm = HuggingFacePipeline(pipeline=generate_text)


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

Answer all questions related to buying a house in Malaysia:
"""
memory = ConversationBufferMemory(memory_key="chat_history")
# qna_prompt = PromptTemplate(input_variables=["chat_history"], template=qna_template)
vectorstore = Weaviate(client,"Qna", "answer", "question")
qna_qachain = RetrievalQA.from_chain_type(llm,chain_type="stuff", retriever=vectorstore.as_retriever(), memory= memory)



#general wikipedia chain 
malaysia_retriever = WikipediaRetriever(top_k_results = 2)
template = """You are a nice chatbot having a conversation with a human.

Chat History:
{chat_history}

New human question: {question}
Answer all questions related to Malaysia, for example history, population, economy, public transport etc:
Response:"""

malaysia_prompt = PromptTemplate.from_template(template)
malaysia_qachain = RetrievalQA.from_chain_type(llm, retriever=malaysia_retriever, memory=memory)

#general places chain
search = GoogleSerperAPIWrapper(type="places", gl="my", k = 5)
result = search.results(query)
print(result)

prompt = PromptTemplate(
    input_variables=["result","chat_history"],
    template="""This is a conversation between a human and a bot:

    {chat_history}
    Given a list of shops or amenities or public transportations,{result}, summarize and output in sentence form.
    
"""
)
#function
chain = LLMChain(llm=llm, prompt=prompt,memory=memory)
print(chain.run({
    'result': result
    }))


# def run_generalqa_chain(query):
#     return qna_qachain(query)



    
#for testing
# def run_qa_chain(query):
#     output = qachain.run(query)
#     print(output)
# print(llm)
# print("test1 llm chatbot\n")
# user_input = input("Please enter your question: \n")

# while user_input != "0": 
#     run_qa_chain(user_input)
#     user_input = input("Please enter your question: \n")

# sys.exit("Exit Program")     



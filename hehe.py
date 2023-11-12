tokenizer = AutoTokenizer.from_pretrained("togethercomputer/GPT-NeoXT-Chat-Base-20B")


WEAVIATE_URL= "https://generalqna-bd3tag0a.weaviate.network"

client = weaviate.Client(
    url = WEAVIATE_URL,
    auth_client_secret=weaviate.AuthApiKey(api_key="JN0qlz84UGUJekopBsrX5xxJBFcwkCJqV9qe"),  # Replace w/ your Weaviate instance API key
    additional_headers = {
        "X-HuggingFace-Api-Key": "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"  # Replace with your inference API key
    }
)


prompt_template = '''

Follow the instructions:
You are a friendly chatbot assistant that responds in a conversational manner to users' questions.
You have knowledge on buying houses in Malaysia.
Explain your answer.
Answer users questions based on your knowledge and our older conversation. Do not make up answers.
If you do not know the answer to a question, just say "I am not sure".


Given the following conversation and a follow up question, answer the question.


Context: {context}


'''

qna_prompt = PromptTemplate(
            template=prompt_template, input_variables=["context"]
        )

general_chain = ConversationalRetrievalChain.from_llm(
                        llm=llm,
                        chain_type="stuff",
                        retriever=vectorstore.as_retriever(),
                        memory=memory,
                        combine_docs_chain_kwargs={'prompt':qna_prompt}
                    )

greetings = ["hello", "hi", "hey", "greetings"]
if any(greeting in query.lower() for greeting in greetings):
    greeting_response = "Hello! I am your all-in-one Property Advisor, Reali AI! How can I assist you today?"
    print(greeting_response)
else:
    # If the user's query is not a greeting, proceed with the conversational retrieval chain
    response = general_chain(query)
    print(response)

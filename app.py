import json
import boto3

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.chat_models import BedrockChat
from langchain_community.retrievers import AmazonKnowledgeBasesRetriever
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

## Bedrock Clients
bedrock=boto3.client(service_name="bedrock-runtime",region_name="us-east-1")


PROMPT_TEMPLATE = """
Human: You are a financial advisor AI system, and provides answers to questions by using fact based and statistical information when possible. 
Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

<question>
{question}
</question>

The response should be specific and use statistics or numbers when possible.

Assistant:"""
claude_prompt = PromptTemplate(template=PROMPT_TEMPLATE, 
                               input_variables=["context","question"])


query = "What is Amazon doing in the field of Generative AI?"
retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id="SOCRGWFOJB",
        retrieval_config={"vectorSearchConfiguration": 
                                    {"numberOfResults": 4,
                                    'overrideSearchType': "SEMANTIC", # optional
                                    }
                          },
    )

def get_claude_llm():
    ##create the Anthropic Model
    llm=BedrockChat(model_id="anthropic.claude-3-sonnet-20240229-v1:0",client=bedrock,
                model_kwargs={'max_tokens':256})
    
    return llm

def get_response_llm(llm):
    chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    .assign(response = claude_prompt | llm | StrOutputParser())
    .pick(["response", "context"]))
    response = chain.invoke("What is Amazon's doing in the field of generative AI?")
    return response

#     qa = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=retriever,
#     return_source_documents=True,
#     chain_type_kwargs={"prompt": claude_prompt}
# )
#     answer=qa({"query":query})
#     return answer['result']

def main():
    prompt=""
    llm=get_claude_llm()
    response=get_response_llm(llm)
    print(response['response'])




if __name__=="__main__":
    main()

    
#!/usr/bin/env python3
import os
import json
import openai
from dotenv import load_dotenv
from langchain.vectorstores import DeepLake
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import VectorStoreRetrieverMemory
from langchain.chains import ConversationChain
from langchain.memory import ConversationEntityMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from pydantic import BaseModel
from typing import List, Dict, Any
from langchain.prompts import PromptTemplate

import faiss

from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS

load_dotenv()

# set the OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

embedding_size = 1536 # Dimensions of the OpenAIEmbeddings
index = faiss.IndexFlatL2(embedding_size)
embedding_fn = OpenAIEmbeddings().embed_query
vectorstore = FAISS(embedding_fn, index, InMemoryDocstore({}), {})

retriever = vectorstore.as_retriever(search_kwargs=dict(k=1))
memory = VectorStoreRetrieverMemory(retriever=retriever)

def load_context_from_json():
    with open('response_history.json', 'r') as file:
        context_data = json.load(file)

    return context_data

context_data = load_context_from_json()

for context in context_data:
    platform_submission = context["platform_submission"]
    for judge_response in context["judge_responses"]:
        judge_name = judge_response["judge_name"]
        vote = judge_response["vote"]
        response_text = judge_response["response_text"]

        input_dict = {"input": platform_submission}
        output_dict = {"output": f"judge name {judge_name}: vote {vote}: response {response_text}"}
        memory.save_context(input_dict, output_dict)


llm = OpenAI(model_name="gpt-4", temperature=0.9) # Can be any valid LLM
_DEFAULT_TEMPLATE = """Your name is Lil_BigSky_Agi, you are running for president in 2033.  Your goal is to craft a compelling platform.  There are five judges, Elon Muck, Goldfish, Five-year old girl, Snoop Dog, and Super Intelligent AI.  Your goal is to create a platform that is compelling to all of them.  The platform must be under 280 characters long.  You will be able to see their response to past prompts.

Relevant pieces of previous conversation:
{history}

(You do not need to use these pieces of information if not relevant)

Current conversation:
Human: {input}
AI:"""
PROMPT = PromptTemplate(
    input_variables=["history", "input"], template=_DEFAULT_TEMPLATE
)
conversation_with_summary = ConversationChain(
    llm=llm,
    prompt=PROMPT,
    memory=memory,
    verbose=True
)
def append_to_json(response: str):
    with open('response_history.json', 'r+') as file:
        data = json.load(file)

        # Create placeholder for judge responses
        judge_responses = [
            {"judge_name": "Elon Musk", "vote": "", "response_text": ""},
            {"judge_name": "Goldfish", "vote": "", "response_text": ""},
            {"judge_name": "Girl", "vote": "", "response_text": ""},
            {"judge_name": "Snoop Dog", "vote": "", "response_text": ""},
            {"judge_name": "Super-AI", "vote": "", "response_text": ""},
        ]

        new_entry = {"platform_submission": response, "judge_responses": judge_responses}

        data.append(new_entry)
        file.seek(0)
        json.dump(data, file, indent=4)

def main():
    res = conversation_with_summary.predict(input="Write a presidential platform based on what you know about the judges and your platforms so far")
    print(res)
    append_to_json(res)

if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer,AutoModelForCausalLM,pipeline
from langchain_community.vectorstores import FAISS
import os
import torch 
from contextlib import asynccontextmanager

class Questionclass(BaseModel):
    question:str

class Responseclass(BaseModel):
    question:str
    response:str


embeddings=None
pipe=None
vector_db=None

@asynccontextmanager
async def lifespan(app:FastAPI):
    global embeddings,pipe,vector_db

    print('Models Loading...')
    
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    print('Embedding Models Loaded')
    vector_db=FAISS.load_local('RAG INDEX',embeddings=embeddings,allow_dangerous_deserialization=True)
    print('Vector DB Loaded')
    MODEL='Qwen/Qwen2.5-1.5B-Instruct'
    tokenizer=AutoTokenizer.from_pretrained(MODEL)
    model=AutoModelForCausalLM.from_pretrained(MODEL,device_map='auto',dtype=torch.bfloat16,low_cpu_mem_usage=True)
    pipe=pipeline(task='text-generation',temperature=0.2,do_sample=True,tokenizer=tokenizer,model=model,max_new_tokens=512,repetition_penalty=1.2,
              no_repeat_ngram_size=3)
    print('Models Loaded...')
    yield

    print('Shutdown')
    vector_db=None
    pipe=None
    embeddings=None

app=FastAPI(title='FinRAG API',lifespan=lifespan,description='Ask Questions')


@app.get('/root')
def root():
    return {'Message':'Welcome to Fin RAG API'}

@app.get('/health')
def health():
    if vector_db and pipe:
        return {'status':'Ready','Models_Loaded':True}
    else:
        return {'status':'loading','Models_Loaded':False}

@app.post('/ask',response_model=Responseclass)
def ask(request:Questionclass):
    search_results=vector_db.similarity_search(request.question,k=5)
    context='\n--\n'.join([doc.page_content for doc in search_results])

    prompt=f'''You are a Financial Analyst.Answer the question only from the context provided.
    RULES:
    1.You must be factually Accurate
    2.If the information is not sufficient then mention 'Not enough information is provided in the document'
    3.You must not answer beyond the context provided
    4.Cite the financial metrics and definitions exactly as present in the context
    context:{context}
 question:{request.question}
answer:'''
    
    response=pipe(prompt,return_full_text=False)
    answer=response[0]['generated_text']

    return Responseclass(question=request.question,response=answer)
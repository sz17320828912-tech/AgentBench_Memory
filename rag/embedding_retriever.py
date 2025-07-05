import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Dict, Union
from openai import OpenAI
import torch.nn.functional as F
from tasks.eval_data_utils import (
    format_chat,
)
import time
import re

from langchain_openai import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Create a custom embedding class for Contriever
class ContrieverEmbeddings(Embeddings):
    def __init__(self, model_name="facebook/contriever"):
        assert "contriever" in model_name, "Model name must contain 'contriever'"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, device_map="auto")
        self.model.eval()
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            embedding = outputs.last_hidden_state[:, 0, :]
            embedding = F.normalize(embedding, p=2, dim=1)
            embeddings.append(embedding.cpu().numpy()[0].tolist())
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.model.device)        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        embedding = outputs.last_hidden_state[:, 0, :]
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding.cpu().numpy()[0].tolist()


class  TextRetriever:
    def __init__(self, embedding_model_name="text-embedding-3-large", sub_dataset=None):
        
        if embedding_model_name == "facebook/contriever":
            self.embedding_model = ContrieverEmbeddings(model_name=embedding_model_name)
        elif embedding_model_name == "text-embedding-3-large" or embedding_model_name == "text-embedding-3-small":
            self.embedding_model = OpenAIEmbeddings(
                                    model=embedding_model_name,
                                    # With the `text-embedding-3` class
                                    # of models, you can specify the size
                                    # of the embeddings you want returned.
                                    # dimensions=1024
                                )
        self.sub_dataset = sub_dataset
        self.vectorstore: FAISS = None
        self._current_documents = None
    
    def build_vectorstore(self, documents: List[str]):
        """Build and cache the vector store from documents"""
        # Convert strings to Document objects if needed
        if isinstance(documents[0], str):
            doc_objects = [Document(page_content=doc) for doc in documents]
        else:
            doc_objects = documents
            
        self.vectorstore = FAISS.from_documents(doc_objects, self.embedding_model)
        self._current_documents = documents
        
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve most relevant contexts for a query (auto-caches vectorstore)"""
    
        
        # Perform similarity search to get top_k most relevant texts
        results = self.vectorstore.similarity_search(query, k=top_k)
        
        # Extract and return the relevant texts
        return [doc.page_content for doc in results]
    



class RAGSystem:
    def __init__(self, retriever, model, temperature, max_tokens):
        self.retriever = retriever
        self.llm = OpenAI()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def answer_query(self, query: str, top_k: int, system_message: str, retrieval_template: str) -> Dict[str, Union[str, float]]:
        """Retrieve relevant information and generate an answer"""
        # Retrieve relevant passages
        start_time = time.time()
        match = re.search(r"Now Answer the Question:\s*(.*)", query, re.DOTALL)
        if match:
            retrieval_query =  ''.join(match.groups())
        else:
            match = re.search(r"Here is the conversation:\s*(.*)", query, re.DOTALL)
            if match:
                retrieval_query =  ''.join(match.groups())
            else:
                retrieval_query = query
        #print(f"Retrieve query: {retrieval_query}")
        retrieved_contexts = self.retriever.retrieve(retrieval_query, top_k)
        
        # Format retrieved contexts
        formatted_context = "\n\n".join([f"Passage {i+1}:\n{text}" 
                                       for i, text in enumerate(retrieved_contexts)])
        memory_construction_time = time.time() - start_time
        
        # Generate prompt
        retrieval_memory_string = "\n".join([f"Memory {i+1}:\n{text}" for i, text in enumerate(retrieved_contexts)])
        templated_message = retrieval_template.format(memory=retrieval_memory_string)
        ask_llm_message=templated_message + "\n" + query
        format_message = format_chat(message=ask_llm_message, system_message=system_message)
        
        # Get response from LLM
        response = self.llm.chat.completions.create(
                                model=self.model,
                                messages=format_message,
                                temperature=self.temperature,
                                max_tokens=self.max_tokens
                            )
        query_time_len = time.time() - start_time - memory_construction_time
        
        return {
            "query": query,
            "context_used": formatted_context,
            "answer": response.choices[0].message.content,
            "memory_construction_time": memory_construction_time,
            "query_time_len": query_time_len,
        }
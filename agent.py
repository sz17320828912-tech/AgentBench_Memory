import os
import json
import torch
import tiktoken
from openai import OpenAI
from tasks.templates import get_template
from tasks.eval_data_utils import (
    format_chat,
)
import re
import time

from langchain_core.documents import Document
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig


class AgentWrapper:
    """
    A wrapper class for different types of memory agents including:
    - Long context agents (GPT, Claude, Gemini)
    - Letta agents
    - Mem0 agents  
    - Cognee agents
    - RAG agents (various implementations)
    """

    def __init__(self, agent_config, dataset_config, load_agent_from):
        """
        Initialize the agent wrapper with specified configuration.
        
        Args:
            agent_config: Configuration dictionary for the agent
            dataset_config: Configuration dictionary for the dataset
            load_agent_from: Optional path to load existing agent state from
        """
        # Basic agent configuration
        self.agent_name = agent_config['agent_name']
        self.sub_dataset = dataset_config['sub_dataset']
        self.context_max_length = dataset_config['context_max_length']
        
        # Output and storage configuration
        self.output_dir = agent_config['output_dir']
        self.agent_save_to_folder = load_agent_from
        
        # Context and token limits
        self.input_length_limit = (agent_config['input_length_limit'] - 
                                 agent_config['buffer_length'] - 
                                 dataset_config['generation_max_length'])
        
        # Model configuration
        self.model = agent_config['model']
        self.max_tokens = dataset_config['generation_max_length']
        self.temperature = agent_config.get('temperature', 0.0)
        
        # Initialize tokenizer (default to gpt-4o-mini for non-gpt models)
        model_for_tokenizer = self.model if "gpt-4o" in self.model else "gpt-4o-mini"
        self.tokenizer = tiktoken.encoding_for_model(model_for_tokenizer)
        
        # Initialize agent based on type
        self._initialize_agent_by_type(agent_config, dataset_config)

    def _initialize_agent_by_type(self, agent_config, dataset_config):
        """Initialize the specific agent type based on agent name."""
        
        if 'Long_context_agent' in self.agent_name:
            self._initialize_long_context_agent()
        elif self._is_agent_type("letta"):
            self._initialize_letta_agent(agent_config)
        elif self._is_agent_type("mem0"):
            self._initialize_mem0_agent(agent_config, dataset_config)
        elif self._is_agent_type("cognee"):
            self._initialize_cognee_agent(agent_config, dataset_config)
        elif self._is_agent_type("rag"):
            self._initialize_rag_agent(agent_config, dataset_config)
        else:
            raise NotImplementedError(f"Agent type not supported: {self.agent_name}")

    def _is_agent_type(self, agent_type):
        """Check if the current agent is of a specific type."""
        return agent_type in self.agent_name

    def _create_standard_response(self, output, input_tokens, output_tokens, memory_time, query_time):
        """Create standardized response dictionary."""
        return {
            "output": output,
            "input_len": input_tokens,
            "output_len": output_tokens,
            "memory_construction_time": memory_time,
            "query_time_len": query_time,
        }

    def _initialize_long_context_agent(self):
        """Initialize long context agent with appropriate client."""
        self.context = ''
        
        if "gpt" in self.model or "o4" in self.model:
            self.client = OpenAI()
        elif "claude" in self.model:
            import anthropic
            self.client = anthropic.Anthropic(
                api_key=os.environ.get('Anthropic_API_KEY'),
            )
        elif "gemini" in self.model:
            from google import genai
            self.client = genai.Client(api_key=os.environ.get('Google_API_KEY'))
        else:
            raise NotImplementedError(f"Model not supported for long context agent: {self.model}")

    def _initialize_letta_agent(self, agent_config):
        """Initialize Letta agent with proper configuration."""
        from letta import create_client, LLMConfig, EmbeddingConfig, BasicBlockMemory

        self.chunk_size = agent_config['agent_chunk_size']
        self.letta_mode = agent_config['letta_mode']
        
        self.client = create_client()
        self.client.set_default_llm_config(LLMConfig.default_config(agent_config['model'])) 
        self.agent_start_time = time.time()
        
        # Configure embedding
        if agent_config['text_embedding'] == 'text-embedding-3-small':
            self.client.set_default_embedding_config(EmbeddingConfig(
                embedding_model="text-embedding-3-small",
                embedding_endpoint_type="openai",
                embedding_endpoint="https://api.openai.com/v1",
                embedding_dim=1536,
                embedding_chunk_size=self.chunk_size,
            ))
        else:
            self.client.set_default_embedding_config(
                EmbeddingConfig.default_config(agent_config['text_embedding'])
            )

        # Load system prompt
        system_path = agent_config['system_path']
        with open(system_path, 'r') as f:
            self.system = f.read()

        # Load or create agent
        if os.path.exists(self.agent_save_to_folder):
            self.load_agent()
        else:
            human_block = self.client.create_block(
                label='human', 
                value='User is sharing the contents they are reading recently.', 
                limit=2000
            )
            persona_block = self.client.create_block(
                label='persona', 
                value='You are a helpful assistant that can help memorize details in the conversation.', 
                limit=2000
            )
            memory = BasicBlockMemory(blocks=[human_block, persona_block])
            self.agent_state = self.client.create_agent(
                name='mm_agent',
                memory=memory,
                system=self.system
            )

    def _initialize_mem0_agent(self, agent_config, dataset_config):
        """Initialize Mem0 agent with retrieval configuration."""
        from mem0.memory.main import Memory
        
        self.retrieve_num = agent_config['retrieve_num']
        self.context = ''
        self.client = OpenAI()
        self.memory = Memory()
        self.agent_start_time = time.time()

    def _initialize_cognee_agent(self, agent_config, dataset_config):
        """Initialize Cognee agent with knowledge graph configuration."""
        self.context = ''
        self.chunks = []
        self.retrieve_num = agent_config['retrieve_num']
        self.chunk_size = agent_config['agent_chunk_size']
        self.agent_start_time = time.time()
        self.cognee_dir = './cognee/.cognee_system/databases/cognee.lancedb'

    def _initialize_rag_agent(self, agent_config, dataset_config):
        """Initialize RAG agent with retrieval configuration."""
        self.context = ''
        self.chunks = []
        self.retrieve_num = agent_config['retrieve_num']
        self.chunk_size = dataset_config['chunk_size']
        self.context_len = 0
        self.context_id = -1

    def send_message(self, message, memorizing=False, query_id=None, context_id=None):
        """
        Send a message to the agent for either memorization or querying.
        
        Args:
            message: The message content (context for memorization, query for answering)
            memorizing: Whether to memorize the message (True) or answer it (False)
            query_id: Unique identifier for the query
            context_id: Unique identifier for the context
            
        Returns:
            dict or str: Agent response with metadata (for queries) or confirmation (for memorization)
        """
        # Route to appropriate agent handler based on agent type
        if 'Long_context_agent' in self.agent_name:
            return self._handle_long_context_agent(message, memorizing)
        elif any(self._is_agent_type(agent_type) for agent_type in ["letta", "cognee", "mem0"]):
            return self._handle_memory_agent(message, memorizing, query_id, context_id)
        elif self._is_agent_type("rag"):
            return self._handle_rag_agent(message, memorizing, query_id, context_id)
        else:
            raise NotImplementedError(f"Agent type not supported: {self.agent_name}")

    def _handle_long_context_agent(self, message, memorizing):
        """Handle message processing for long context agents."""
        if memorizing:
            # Add message to context memory
            memorize_template = get_template(self.sub_dataset, 'memorize', self.agent_name)
            formatted_message = memorize_template.format(context=message)
            self.context += "\n" + formatted_message
            self.context = self.context.strip()
            return "Memorized"
        else:
            # Process query with context
            return self._query_long_context_agent(message)

    def _query_long_context_agent(self, message):
        """Process a query for long context agents."""
        # Get appropriate tokenizer
        try:
            tokenizer = tiktoken.encoding_for_model(self.model)
        except:
            tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
        
        # Handle context truncation for non-long context models
        buffer_length = 50000
        if self.input_length_limit <= self.context_max_length + buffer_length:
            self._truncate_context_if_needed(tokenizer)
                
        # Format message with context and system prompt
        retrieval_memory = get_template(self.sub_dataset, 'retrieval', self.agent_name)
        retrieval_memory = retrieval_memory.format(memory=self.context)
        full_message = retrieval_memory + "\n" + message
        system_message = get_template(self.sub_dataset, 'system', self.agent_name)
        formatted_message = format_chat(message=full_message, system_message=system_message)
        
        # Query the model
        start_time = time.time()
        
        if "gpt" in self.model: 
            response = self.client.chat.completions.create(
                model=self.model,
                messages=formatted_message,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return self._format_openai_response(response, start_time)
            
        elif "o4" in self.model:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=formatted_message,
            )
            return self._format_openai_response(response, start_time)
            
        elif "claude" in self.model:
            return self._query_claude(full_message, system_message, start_time)
            
        elif "gemini" in self.model:
            return self._query_gemini(formatted_message, start_time)
            
        else:
            raise NotImplementedError(f"Model not supported: {self.model}")

    def _truncate_context_if_needed(self, tokenizer):
        """Truncate context if it exceeds limits."""
        # Truncate context if it exceeds the context_max_length
        if len(tokenizer.encode(self.context, disallowed_special=())) > self.context_max_length:
            encoded = tokenizer.encode(self.context, disallowed_special=())
            self.context = tokenizer.decode(encoded[-self.context_max_length:])
        
        # Truncate if context exceeds the input_length_limit
        if len(tokenizer.encode(self.context, disallowed_special=())) > self.input_length_limit:
            encoded = tokenizer.encode(self.context, disallowed_special=())
            self.context = tokenizer.decode(encoded[-self.input_length_limit:])

    def _format_openai_response(self, response, start_time):
        """Format OpenAI API response into standard output format."""
        return self._create_standard_response(
            response.choices[0].message.content,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            0,
            time.time() - start_time
        )

    def _query_claude(self, message, system_message, start_time):
        """Query Claude model with proper formatting."""
        formatted_message = format_chat(message=message, system_message=system_message, include_system=False)
        response = self.client.messages.create(
            model=self.model,
            messages=formatted_message,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return self._create_standard_response(
            response.content[0].text,
            response.usage.input_tokens,
            response.usage.output_tokens,
            0,
            time.time() - start_time
        )

    def _query_gemini(self, formatted_message, start_time):
        """Query Gemini model with proper configuration."""
        from google.genai import types
        response = self.client.models.generate_content(
            model=self.model,
            contents=formatted_message[1]["content"],
            config=types.GenerateContentConfig(
                system_instruction=formatted_message[0]["content"], 
                temperature=self.temperature,
                max_output_tokens=self.max_tokens
            )      
        )
        return self._create_standard_response(
            response.text,
            response.usage_metadata.prompt_token_count,
            response.usage_metadata.candidates_token_count,
            0,
            time.time() - start_time
        )
    def _handle_memory_agent(self, message, memorizing, query_id, context_id):
        """Handle message processing for memory-based agents (Letta, Cognee, Mem0)."""
        if self._is_agent_type("letta"):
            return self._handle_letta_agent(message, memorizing, query_id, context_id)
        elif self._is_agent_type("cognee"):
            return self._handle_cognee_agent(message, memorizing, query_id, context_id)
        elif self._is_agent_type("mem0"):
            return self._handle_mem0_agent(message, memorizing, query_id, context_id)
        else:
            raise NotImplementedError(f"Memory agent type not supported: {self.agent_name}")

    def _handle_letta_agent(self, message, memorizing, query_id, context_id):
        """Handle message processing for Letta agents."""
        # Format message based on context
        formatted_message = (get_template(self.sub_dataset, 'memorize', self.agent_name).format(context=message) 
                           if memorizing else message)
        
        # Handle memory construction time for queries
        memory_construction_time = 0 if memorizing else time.time() - self.agent_start_time
        
        # Reload agent for queries
        if not memorizing:
            if os.path.exists(self.agent_save_to_folder):
                self.load_agent()
            else:
                print(f"\n\nAgent {self.agent_name} not found in {self.agent_save_to_folder}\n\n")
        
        # Process based on Letta mode
        response = self._process_letta_message(formatted_message, memorizing)
        
        if memorizing:
            return "Memorized"
        
        # Create response for queries
        tokenizer = self.tokenizer
        query_time_len = time.time() - self.agent_start_time - memory_construction_time
        output = self._create_standard_response(
            response,
            len(tokenizer.encode(message, disallowed_special=())),
            len(tokenizer.encode(response, disallowed_special=())),
            memory_construction_time,
            query_time_len
        )
        self.agent_start_time = time.time()  # Reset time
        return output
    
    def _process_letta_message(self, formatted_message, memorizing):
        """Process message with Letta client based on mode."""
        try:
            if self.letta_mode == 'insert':
                if memorizing:
                    self.client.server.passage_manager.insert_passage(
                        agent_state=self.agent_state,
                        agent_id=self.agent_state.id,
                        text=formatted_message,
                        actor=self.client.user,
                    )
                    return "Memorized"
                else:
                    response = self.client.send_message(
                        agent_id=self.agent_state.id,
                        message=formatted_message,
                        role='user')
                    return json.loads(response.messages[-3].tool_call.arguments)['message']
            
            elif self.letta_mode == 'chat':
                response = self.client.send_message(
                    agent_id=self.agent_state.id,
                    message=formatted_message,
                    role='user')
                
                if memorizing:
                    return "Memorized"
                else:
                    return json.loads(response.messages[-3].tool_call.arguments)['message']
        except Exception as e:
            return f"{e}"

    def _handle_cognee_agent(self, message, memorizing, query_id, context_id):
        """Handle message processing for Cognee agents."""
        import cognee
        import asyncio
        dataset_name = f'default_dataset_{self.sub_dataset}_context_{context_id}'
        
        if memorizing:
            # Add context to Cognee knowledge base
            formatted_message = get_template(self.sub_dataset, 'memorize', self.agent_name).format(context=message)
            
            # Add text to cognee and generate knowledge graph
            asyncio.run(cognee.add(formatted_message, dataset_name=dataset_name))
            asyncio.run(cognee.cognify(datasets=[dataset_name], chunk_size=self.chunk_size))

            self.context += "\n" + formatted_message
            self.context = self.context.strip()
            return "Memorized"
        else:                    
            # Query the knowledge graph
            memory_construction_time = time.time() - self.agent_start_time
            searched_results = asyncio.run(cognee.search(
                query_text=message, 
                top_k=self.retrieve_num, 
                datasets=[dataset_name]
            ))
                    
            # Format results
            total_results = ("".join([f"{result}\n" for result in searched_results]) 
                           if searched_results else "No results found.")
            
            # Return formatted output
            tokenizer = self.tokenizer
            query_time_len = time.time() - self.agent_start_time - memory_construction_time
            output = self._create_standard_response(
                total_results,
                len(tokenizer.encode(self.context, disallowed_special=())),
                len(tokenizer.encode(total_results, disallowed_special=())),
                memory_construction_time,
                query_time_len
            )
            self.agent_start_time = time.time()  # Reset time
            return output

    def _handle_mem0_agent(self, message, memorizing, query_id, context_id):
        """Handle message processing for Mem0 agents."""
        user_id = f'context_{context_id}_{self.sub_dataset}'
        if memorizing:
            system_message = get_template(self.sub_dataset, 'system', self.agent_name)
            formatted_message = get_template(self.sub_dataset, 'memorize', self.agent_name).format(context=message)
            
            # Generate Assistant response
            # memory_messages = [{"role": "system", "content": system_message}, {"role": "user", "content": formatted_message}]
            # response = OpenAI().chat.completions.create(
            #             model=self.model,
            #             messages=memory_messages,
            #             max_tokens=1000,
            #         )
            # memory_messages = [
            #     {"role": "system", "content": system_message}, 
            #     {"role": "user", "content": formatted_message},
            #     {"role": "assistant", "content": response.choices[0].message.content}
            # ]
            memory_messages = [
                {"role": "system", "content": system_message}, 
                {"role": "user", "content": formatted_message},
                {"role": "assistant", "content": "I'll make sure to add the content into the memory."}
            ]
            
            vector_results = self.memory.add(memory_messages, user_id=user_id)
            print(f"\n\n\nvector_results: {vector_results}\n\n\n")
            return "Memorized"
        else:
            # Retrieve relevant memories and generate response
            memory_construction_time = time.time() - self.agent_start_time
            relevant_memories = self.memory.search(query=message, user_id=user_id, limit=self.retrieve_num)
            print(f"\n\n\nrelevant_memories: {relevant_memories}\n\n\n")
            
            memories_str = "\n".join(f"- {entry['memory']}" for entry in relevant_memories["results"])
            
            # Generate assistant response
            retrieval_message = get_template(self.sub_dataset, 'retrieval', self.agent_name).format(memory=memories_str)
            system_prompt = f"You are a helpful AI. Answer the question based on query and memories.\n{retrieval_message}\n"
            llm_messages = [
                {"role": "system", "content": system_prompt}, 
                {"role": "user", "content": message}
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=llm_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            memory_retrieval_length = len(self.tokenizer.encode(memories_str, disallowed_special=()))
            query_time_len = time.time() - self.agent_start_time - memory_construction_time
            print(f"\nmemory_length: {memory_retrieval_length}\n")
            
            output = self._create_standard_response(
                response.choices[0].message.content,
                response.usage.prompt_tokens + memory_retrieval_length,
                response.usage.completion_tokens,
                memory_construction_time,
                query_time_len
            )
            self.agent_start_time = time.time()  # Reset time
            return output

    def _handle_rag_agent(self, message, memorizing, query_id, context_id):
        """Handle message processing for RAG agents."""
        if memorizing:
            # Add message to chunks and context
            self.context += "\n" + message
            self.context = self.context.strip()
            self.chunks.append(message)
            self.context_len = self.context_len + self.chunk_size   
            
            # Truncate context if it exceeds limits
            if self.context_len > self.input_length_limit:
                self.chunks = self.chunks[1:]
                self.context_len = self.context_len - self.chunk_size
            return ''
        else:
            # Handle query processing for different RAG types
            return self._process_rag_query(message, query_id, context_id)

    def _process_rag_query(self, message, query_id, context_id):
        """Process query for RAG agents with different retrieval strategies."""
                
        # Truncate context if needed
        tokenizer = self.tokenizer
        if len(tokenizer.encode(self.context, disallowed_special=())) > self.input_length_limit:
            encoded = tokenizer.encode(self.context, disallowed_special=())
            self.context = tokenizer.decode(encoded[-self.input_length_limit:])
        if self.context_len > self.input_length_limit:
            self.chunks = self.chunks[1:]
            self.context_len = self.context_len - self.chunk_size
        
        # Route to specific RAG implementation and get result
        rag_handlers = {
            "graph_rag": lambda: self._handle_graph_rag(message, context_id, tokenizer),
            "hippo_rag_v2_nv": lambda: self._handle_hippo_rag(message, context_id, tokenizer),
            "hippo_rag_v2_openai": lambda: self._handle_hippo_rag(message, context_id, tokenizer),
            "rag_bm25": lambda: self._handle_bm25_rag(message, context_id, tokenizer),
            "rag_contriever": lambda: self._handle_embedding_rag(message, context_id, tokenizer),
            "rag_text_embedding_3_large": lambda: self._handle_embedding_rag(message, context_id, tokenizer),
            "rag_text_embedding_3_small": lambda: self._handle_embedding_rag(message, context_id, tokenizer),
            "rag_raptor": lambda: self._handle_raptor_rag(message, context_id, tokenizer),
            "rag_nv_embed_v2": lambda: self._handle_nv_embed_rag(message, query_id, context_id, tokenizer),
            "self_rag": lambda: self._handle_self_rag(message, context_id, tokenizer),
        }
        
        # Find matching handler
        handler = next((handler for agent_type, handler in rag_handlers.items() if self._is_agent_type(agent_type)), None)
        if not handler:
            raise NotImplementedError(f"RAG agent type not supported: {self.agent_name}")
        
        output = handler()

        # Save the retrieved context as JSON (if the method provides it)
        if output.get("retrieval_context"):
            save_dir = f"./outputs/rag_retrieved/{self.agent_name}/k_{self.retrieve_num}/{self.sub_dataset}/chunksize_{self.chunk_size}/query_{query_id}_context_{context_id}.json"
            os.makedirs(os.path.dirname(save_dir), exist_ok=True)
            with open(save_dir, "w") as f:
                json.dump(output["retrieval_context"], f)
            
            # drop the retrieval_context       
            output.pop("retrieval_context")
        
        return output

    def _handle_graph_rag(self, message, context_id, tokenizer):
        """Handle Graph RAG processing."""
        start_time = time.time()

        # Build vectorstore if context changed
        if self.context_id != context_id:
            docs = [Document(page_content=t, metadata={"source":"Not provided", "chunk":i}) for i,t in enumerate(self.chunks)]
            try:
                from rag.graph_rag import GraphRAG
                self.graph_rag = GraphRAG(temperature=self.temperature, model_name=self.model, retrieve_num=self.retrieve_num, max_tokens=self.max_tokens)
                self.graph_rag.process_documents(docs)
                memory_construction_time = time.time() - start_time
            except Exception as e:
                print(f"\n\n\n\nError: {e}\n\n\n\n")
            print(f"\n\nGraph RAG build vectorstore finished...\n\n")
        else:
            memory_construction_time = 0
            print(f"\n\nContext {context_id} already processed, skipping Graph RAG build vectorstore...\n\n")

        # Process query
        try:
            response, retrieval_context = self.graph_rag.query(query=message)
        except Exception as e:
            response = f"{e}"
            retrieval_context = "ERROR"
            print(f"\n\n\n\nError: {e}\n\n\n\n")
        
        self.context_id = context_id
        
        print(f"\n\n\n\nResponse: {response}\n\n\n\n")
        if isinstance(response, str):
            response = response
        else:
            response = response.content
        query_time_len = time.time() - start_time - memory_construction_time
        
        return {
            "output": response,
            "input_len": len(tokenizer.encode(retrieval_context + "\n" + message, disallowed_special=())),
            "output_len": len(tokenizer.encode(response, disallowed_special=())),
            "memory_construction_time": memory_construction_time,
            "query_time_len": query_time_len,
            "retrieval_context": retrieval_context,
        }

    def _handle_hippo_rag(self, message, context_id, tokenizer):
        """Handle HippoRAG processing."""
        start_time = time.time()
        
        if self.context_id != context_id:
            docs = self.chunks
            from rag.hipporag import HippoRAG
            if any(agent_name in self.agent_name for agent_name in ["hippo_rag_v2_nv"]):
                save_dir = os.path.join(f"./outputs/rag_retrieved/NV-Embed-v2", self.sub_dataset, f'chunksize_{self.chunk_size}', f'context_id_{context_id}')
                embedding_model_name = 'nvidia/NV-Embed-v2'
            elif any(agent_name in self.agent_name for agent_name in ["hippo_rag_v2_openai"]):
                save_dir = os.path.join(f"./outputs/rag_retrieved/OpenAIEmbedding", self.sub_dataset, f'chunksize_{self.chunk_size}', f'context_id_{context_id}') 
                embedding_model_name = 'text-embedding-ada-002'
            
            self.hipporag = HippoRAG(save_dir=save_dir,
                                llm_model_name=self.model,
                                embedding_model_name=embedding_model_name) 
            self.hipporag.index(docs=docs)
            memory_construction_time = time.time() - start_time
            print(f"\n\nHippoRAG build vectorstore finished...\n\n")
        else:
            memory_construction_time = 0
            print(f"\n\nContext {context_id} already processed, skipping HippoRAG build vectorstore...\n\n")
            
        # Retrieve and answer
        queries = [message]
        retrieval_results, top_k_docs = self.hipporag.retrieve(queries=queries, num_to_retrieve=self.retrieve_num)
        
        qa_results = self.hipporag.rag_qa(retrieval_results)
        response = qa_results[0][0].answer
        
        retrieval_context = "\n\n".join([f"Passage {i+1}:\n{text}" for i, text in enumerate(top_k_docs)])
        query_time_len = time.time() - start_time - memory_construction_time
        
        self.context_id = context_id
        
        return {
            "output": response,
            "input_len": len(tokenizer.encode(retrieval_context + "\n" + message, disallowed_special=())),
            "output_len": len(tokenizer.encode(response, disallowed_special=())),
            "memory_construction_time": memory_construction_time,
            "query_time_len": query_time_len,
            "retrieval_context": retrieval_context,
        }

    # RAG implementation methods
    def _handle_bm25_rag(self, message, context_id, tokenizer):
        """Handle BM25 RAG processing."""
        start_time = time.time()
        
        # Extract retrieval query from message
        retrieval_query = self._extract_retrieval_query(message)
        
        # Build vectorstore if context changed
        if self.context_id != context_id:
            from langchain_community.retrievers import BM25Retriever
            docs = [Document(page_content=t, metadata={"source":"Not provided", "chunk":i}) for i,t in enumerate(self.chunks)]
            self.bm25_retriever = BM25Retriever.from_documents(docs)
            print(f"\n\nBM25 build vectorstore finished...\n\n")
        else:
            print(f"\n\nContext {context_id} already processed, skipping BM25 build vectorstore...\n\n")
        
        # Retrieve documents
        self.bm25_retriever.k = self.retrieve_num
        bm25_documents = self.bm25_retriever.get_relevant_documents(retrieval_query)   
        retrieval_context = [f"{doc.page_content}\n" for doc in bm25_documents] 
        memory_construction_time = time.time() - start_time
        
        # Answer the query
        retrieval_memory_string = "\n".join([f"Memory {i+1}:\n{text}" for i, text in enumerate(retrieval_context)])
        templated_message = get_template(self.sub_dataset, 'retrieval', self.agent_name).format(memory=retrieval_memory_string)
        
        # Format the message
        ask_llm_message = templated_message + "\n" + message
        system_message = get_template(self.sub_dataset, 'system', self.agent_name)
        format_message = format_chat(message=ask_llm_message, system_message=system_message)
        
        # Generate response
        response = OpenAI().chat.completions.create(
            model=self.model,
            messages=format_message,
            temperature=self.temperature,
            max_tokens=self.max_tokens if "gpt-4" in self.model else None
        )
        
        query_time_len = time.time() - start_time - memory_construction_time
        self.context_id = context_id
        
        return {
            "output": response.choices[0].message.content,
            "input_len": response.usage.prompt_tokens,
            "output_len": response.usage.completion_tokens,
            "memory_construction_time": memory_construction_time,
            "query_time_len": query_time_len,
            "retrieval_context": retrieval_context,
        }
    
    def _extract_retrieval_query(self, message):
        """Extract retrieval query from message using regex patterns."""
        patterns = [
            r"Now Answer the Question:\s*(.*)",
            r"Here is the conversation:\s*(.*)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message, re.DOTALL)
            if match:
                return ''.join(match.groups())
        
        return message
        
    def _handle_embedding_rag(self, message, context_id, tokenizer):
        """Handle embedding-based RAG processing (Contriever, Text-embedding models)."""
        from rag.embedding_retriever import TextRetriever, RAGSystem
        
        # Determine embedding model
        if any(agent_name in self.agent_name for agent_name in ["rag_contriever"]):
            embedding_model_name = "facebook/contriever"
        elif any(agent_name in self.agent_name for agent_name in ["rag_text_embedding_3_large"]):
            embedding_model_name = "text-embedding-3-large"
        elif any(agent_name in self.agent_name for agent_name in ["rag_text_embedding_3_small"]):
            embedding_model_name = "text-embedding-3-small"
        else:
            raise NotImplementedError
        
        # Build vectorstore if context changed
        if self.context_id != context_id:
            self.retriever = TextRetriever(embedding_model_name=embedding_model_name)
            self.retriever.build_vectorstore(self.chunks)
            print(f"\n\n{embedding_model_name} build vectorstore finished...\n\n")
        else:
            print(f"\n\nContext {context_id} already processed, skipping {embedding_model_name} build vectorstore...\n\n")
                            
        # Retrieve relevant passages and answer the query
        rag_system = RAGSystem(self.retriever, self.model, self.temperature, self.max_tokens)
        system_message = get_template(self.sub_dataset, 'system', self.agent_name)
        retrieval_template = get_template(self.sub_dataset, 'retrieval', self.agent_name)
        result = rag_system.answer_query(
            query=message, 
            top_k=self.retrieve_num, 
            system_message=system_message, 
            retrieval_template=retrieval_template
        )
        retrieval_context = result['context_used']
        
        self.context_id = context_id
        
        return {
            "output": result["answer"],
            "input_len": len(tokenizer.encode(retrieval_context + "\n" + message, disallowed_special=())),
            "output_len": len(tokenizer.encode(result["answer"], disallowed_special=())),
            "memory_construction_time": result.get("memory_construction_time", result.get("memory_construction_time", 0)),
            "query_time_len": result["query_time_len"],
            "retrieval_context": retrieval_context,
        }
        
    def _handle_raptor_rag(self, message, context_id, tokenizer):
        """Handle RAPTOR RAG processing."""
        # Build vectorstore if context changed
        if self.context_id != context_id:
            texts = self.chunks
            from rag.raptor import RAPTORMethod
            self.raptor_method = RAPTORMethod(texts, max_levels=3)
            print(f"\n\nRaptor build vectorstore finished...\n\n")
        else:
            print(f"\n\nContext {context_id} already processed, skipping Raptor build vectorstore...\n\n")
        
        # Retrieve relevant passages and answer the query
        result = self.raptor_method.run(query=message, k=self.retrieve_num)
        response = result['answer']
        retrieval_context = result['context_used']
        
        self.context_id = context_id
        
        return {
            "output": response,
            "input_len": len(tokenizer.encode(retrieval_context + "\n" + message, disallowed_special=())),
            "output_len": len(tokenizer.encode(response, disallowed_special=())),
            "memory_construction_time": result.get("memory_construction_time", result.get("memory_construction_time", 0)),
            "query_time_len": result["query_time_len"],
            "retrieval_context": retrieval_context,
        }
        
    def _handle_nv_embed_rag(self, message, query_id, context_id, tokenizer):
        """Handle NV-Embed RAG processing."""
        start_time = time.time()
        
        # Load the retrieved context from hippo_rag_v2_nv (since the embedding model is the same)
        query_dir = os.path.join(
            f"./outputs/rag_retrieved/Structure_rag_hippo_rag_v2_nv", 
            f'k_{self.retrieve_num}', 
            self.sub_dataset, 
            f'chunksize_{self.chunk_size}', 
            f'query_{query_id}_context_{context_id}.json'
        )
        
        with open(query_dir, 'r') as f:
            loaded_context = json.load(f)
            
        memory_construction_time = time.time() - start_time
        
        # Answer the query
        retrieval_template = get_template(self.sub_dataset, 'retrieval', self.agent_name)
        retrieval_message = retrieval_template.format(memory=loaded_context)
        ask_llm_message = retrieval_message + "\n" + message
        system_message = get_template(self.sub_dataset, 'system', self.agent_name)
        format_message = format_chat(message=ask_llm_message, system_message=system_message)
        
        # Generate response
        response = self.client.chat.completions.create(
            model=self.model,
            messages=format_message,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        retrieval_context = loaded_context
        query_time_len = time.time() - start_time - memory_construction_time
        
        return {
            "output": response.choices[0].message.content,
            "input_len": response.usage.prompt_tokens,
            "output_len": response.usage.completion_tokens,
            "memory_construction_time": memory_construction_time,
            "query_time_len": query_time_len,
            "retrieval_context": retrieval_context,
        }
        
    def _handle_self_rag(self, message, context_id, tokenizer):
        """Handle Self-RAG processing."""
        from rag.self_rag import SelfRAG
        start_time = time.time()
        
        # Build vectorstore if context changed
        if self.context_id != context_id:
            docs = [Document(page_content=t, metadata={"source":"Not provided", "chunk":i}) for i,t in enumerate(self.chunks)]
            self.self_rag = SelfRAG(documents=docs, temperature=self.temperature, top_k=self.retrieve_num)
            print(f"\n\nSelf-RAG build vectorstore finished...\n\n")
        else:
            print(f"\n\nContext {context_id} already processed, skipping Self-RAG build vectorstore...\n\n")
        
        # Process query
        try:
            response, retrieval_context_list, memory_construction_time, query_time_len = self.self_rag.run(query=message)
        except Exception as e:
            response = f"{e}"
            retrieval_context_list = ["ERROR"]
            memory_construction_time = 0
            query_time_len = 0
            print(f"\n\n\n\nError: {e}\n\n\n\n")
        
        # Prepare the context
        retrieval_context = "\n\n".join([f"Passage {i+1}:\n{text}" 
                                        for i, text in enumerate(retrieval_context_list)])
        
        self.context_id = context_id
        
        return {
            "output": response,
            "input_len": len(tokenizer.encode(retrieval_context + "\n" + message, disallowed_special=())),
            "output_len": len(tokenizer.encode(response, disallowed_special=())),
            "memory_construction_time": memory_construction_time,
            "query_time_len": query_time_len,
            "retrieval_context": retrieval_context,
        }

    def save_agent(self):
        """Save agent state to disk for persistence."""
        # Currently only implemented for Letta agents
        if not self._is_agent_type("letta"):
            print("\n\n Agent not saved (not implemented for this agent type) \n\n")
            return
        
        agent_save_folder = self.agent_save_to_folder
        os.makedirs(agent_save_folder, exist_ok=True)
        
        import shutil
        # Copy the SQLite database file to the target folder
        source_db_path = os.path.expanduser("~/.letta/sqlite.db")
        target_db_path = f"{agent_save_folder}/sqlite.db"
        shutil.copyfile(source_db_path, target_db_path)
        
        # Save the agent ID for future loading
        with open(f"{agent_save_folder}/agent_id.txt", "w") as f:
            f.write(self.agent_state.id)
        print("\n\n Agent saved...\n\n")

    def load_agent(self):
        """Load agent state from disk."""
        agent_save_folder = self.agent_save_to_folder
        assert os.path.exists(agent_save_folder), f"Folder {agent_save_folder} does not exist."

        if not self._is_agent_type("letta"):
            print("\n\nAgent loading not implemented for this agent type\n\n")
            return None

        import shutil
        # Copy the database file back to the Letta directory
        source_db_path = f"{agent_save_folder}/sqlite.db"
        target_db_path = os.path.expanduser("~/.letta/sqlite.db")
        shutil.copyfile(source_db_path, target_db_path)

        # Load agent ID and find the corresponding agent state
        with open(f"{agent_save_folder}/agent_id.txt", "r") as f:
            agent_id = f.read()

        # Find the agent state with the matching ID
        for agent_state in self.client.list_agents():
            if agent_state.id == agent_id:
                self.agent_state = agent_state
                break
        print("\n\n Agent loaded successfully...\n\n")
        
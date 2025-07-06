import os
import sys
import yaml
import dotenv
import time
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
from argparse import ArgumentParser

from agent import AgentWrapper
from initialization import setup_configs_and_directories
from conversation_creator import ConversationCreator

# Load environment variables
dotenv.load_dotenv()

app = FastAPI(title="MemAgent Server", description="Server for Memory Agent Operations")

# Global agent instance
agent_instance = None
agent_config = None
dataset_config = None


class WriteRequest(BaseModel):
    message: str
    context_id: Optional[int] = None


class QueryRequest(BaseModel):
    message: str
    query_id: Optional[int] = None
    context_id: Optional[int] = None


class ServerResponse(BaseModel):
    success: bool
    data: Any
    error: Optional[str] = None


def initialize_agent_from_configs(context_id: int = 0):
    """Initialize the agent for a new context, clearing previous memories."""
    global agent_instance, agent_config, dataset_config
    
    if agent_config is None or dataset_config is None:
        return False, "Server not configured. Agent and dataset configs must be provided at server startup."
    
    try:
        # Generate agent save folder for this context
        agent_save_folder = f"./agents/{agent_config['agent_name']}_{dataset_config['sub_dataset']}/exp_{context_id}"
        
        # Initialize agent for new context (this will create a fresh agent)
        agent_instance = AgentWrapper(agent_config, dataset_config, load_agent_from=agent_save_folder)
        
        # If agent doesn't exist, it will be ready for memorization
        if not os.path.exists(agent_save_folder):
            print(f"New agent context {context_id} initialized for memorization.")
        else:
            agent_instance.load_agent()
            print(f"Agent loaded from {agent_save_folder}")
            
        return True, f"Agent initialized for context {context_id}"
    except Exception as e:
        return False, f"Failed to initialize agent: {str(e)}"


@app.post("/initialize")
async def initialize_agent(context_id: int = 0):
    """Initialize the agent, clearing memories. Context ID is optional for organizing separate experiments."""
    success, message = initialize_agent_from_configs(context_id)
    
    if success:
        return ServerResponse(success=True, data={"message": message})
    else:
        raise HTTPException(status_code=500, detail=message)


@app.post("/write")
async def write_endpoint(request: WriteRequest):
    """Write/memorize content to the agent."""
    global agent_instance
    
    if agent_instance is None:
        raise HTTPException(status_code=400, detail="Agent not initialized. Call /initialize first.")
    
    try:
        # Call agent's send_message with memorizing=True
        result = agent_instance.send_message(
            message=request.message,
            memorizing=True,
            context_id=request.context_id
        )
        
        return ServerResponse(
            success=True,
            data={"result": result, "message": "Content memorized successfully"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in write operation: {str(e)}")


@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """Query the agent for information."""
    global agent_instance
    
    if agent_instance is None:
        raise HTTPException(status_code=400, detail="Agent not initialized. Call /initialize first.")
    
    try:
        # Call agent's send_message with memorizing=False
        result = agent_instance.send_message(
            message=request.message,
            memorizing=False,
            query_id=request.query_id,
            context_id=request.context_id
        )
        
        return ServerResponse(
            success=True,
            data={"result": result}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in query operation: {str(e)}")


@app.post("/save_agent")
async def save_agent_endpoint():
    """Save the current agent state."""
    global agent_instance
    
    if agent_instance is None:
        raise HTTPException(status_code=400, detail="Agent not initialized. Call /initialize first.")
    
    try:
        agent_instance.save_agent()
        return ServerResponse(
            success=True,
            data={"message": "Agent saved successfully"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving agent: {str(e)}")


@app.post("/load_agent")
async def load_agent_endpoint():
    """Load the agent state."""
    global agent_instance
    
    if agent_instance is None:
        raise HTTPException(status_code=400, detail="Agent not initialized. Call /initialize first.")
    
    try:
        agent_instance.load_agent()
        return ServerResponse(
            success=True,
            data={"message": "Agent loaded successfully"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading agent: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "agent_initialized": agent_instance is not None}


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "MemAgent Server API",
        "endpoints": {
            "/initialize": "POST - Initialize agent with configurations",
            "/write": "POST - Write/memorize content to agent",
            "/query": "POST - Query agent for information",
            "/save_agent": "POST - Save current agent state",
            "/load_agent": "POST - Load agent state",
            "/health": "GET - Health check"
        }
    }


def parse_server_arguments():
    """Parse command line arguments for the server."""
    parser = ArgumentParser()
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to bind the server to'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port to bind the server to'
    )
    parser.add_argument(
        '--reload',
        action='store_true',
        help='Enable auto-reload for development'
    )
    # Configuration arguments
    parser.add_argument(
        '--agent_config', 
        type=str, 
        required=True,
        help='Path to agent configuration file'
    )
    parser.add_argument(
        '--dataset_config', 
        type=str, 
        required=True,
        help='Path to dataset configuration file'
    )
    parser.add_argument(
        '--chunk_size_ablation', 
        type=int, 
        default=0,
        help='Override chunk size for ablation studies (0 = use config default)'
    )
    parser.add_argument(
        '--max_test_queries_ablation', 
        type=int, 
        default=0,
        help='Limit maximum test queries for ablation studies (0 = no limit)'
    )
    parser.add_argument(
        '--force', 
        action='store_true', 
        default=False,
        help='Force re-run even if results already exist'
    )
    return parser.parse_args()


def setup_server_configs(args):
    """Setup server configurations from command line arguments."""
    global agent_config, dataset_config
    
    try:
        # Create a mock args object for setup_configs_and_directories
        class MockArgs:
            def __init__(self, agent_config, dataset_config, chunk_size_ablation, max_test_queries_ablation, force):
                self.agent_config = agent_config
                self.dataset_config = dataset_config
                self.chunk_size_ablation = chunk_size_ablation
                self.max_test_queries_ablation = max_test_queries_ablation
                self.force = force
        
        mock_args = MockArgs(
            args.agent_config, 
            args.dataset_config, 
            args.chunk_size_ablation, 
            args.max_test_queries_ablation,
            args.force
        )
        agent_config, dataset_config, _ = setup_configs_and_directories(mock_args)
        
        print(f"✅ Server configured with:")
        print(f"   Agent: {agent_config['agent_name']}")
        print(f"   Dataset: {dataset_config['dataset']}")
        print(f"   Sub-dataset: {dataset_config['sub_dataset']}")
        if args.chunk_size_ablation > 0:
            print(f"   Chunk size ablation: {args.chunk_size_ablation}")
        if args.max_test_queries_ablation > 0:
            print(f"   Max test queries ablation: {args.max_test_queries_ablation}")
            
        return True
    except Exception as e:
        print(f"❌ Failed to setup server configs: {str(e)}")
        return False


if __name__ == "__main__":
    args = parse_server_arguments()
    
    # Setup configurations
    if not setup_server_configs(args):
        sys.exit(1)
    
    print(f"🚀 Starting MemAgent server on {args.host}:{args.port}")
    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    ) 
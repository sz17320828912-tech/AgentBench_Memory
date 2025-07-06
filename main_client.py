import os
import yaml
import dotenv
import time
import json
import requests
from argparse import ArgumentParser
from conversation_creator import ConversationCreator
from initialization import (
    load_existing_results, 
    create_agent_and_fetch_data, 
    setup_configs_and_directories, 
    generate_agent_save_folder
)
from tqdm import tqdm
from collections import defaultdict
import logging
import numpy as np
from tasks.eval_other_utils import metrics_summarization

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Load environment variables
dotenv.load_dotenv()


class AgentClient:
    """Client for communicating with the MemAgent server."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def initialize_agent(self, context_id: int = 0):
        """Initialize the agent for a new context on the server."""
        url = f"{self.base_url}/initialize"
        params = {"context_id": context_id}
        
        response = self.session.post(url, params=params)
        response.raise_for_status()
        return response.json()
    
    def write_message(self, message: str, context_id: int = None):
        """Send a message to be memorized by the agent."""
        url = f"{self.base_url}/write"
        data = {
            "message": message,
            "context_id": context_id
        }
        
        response = self.session.post(url, json=data)
        response.raise_for_status()
        return response.json()
    
    def query_message(self, message: str, query_id: int = None, context_id: int = None):
        """Send a query to the agent."""
        url = f"{self.base_url}/query"
        data = {
            "message": message,
            "query_id": query_id,
            "context_id": context_id
        }
        
        response = self.session.post(url, json=data)
        response.raise_for_status()
        return response.json()
    
    def endpoint_exists(self, endpoint: str) -> bool:
        """Check if an endpoint exists on the server."""
        try:
            # Try to make a request to the endpoint and see if we get a 404
            url = f"{self.base_url}{endpoint}"
            response = self.session.post(url)
            # If we get anything other than 404, the endpoint exists
            return response.status_code != 404
        except:
            return False
    
    def save_agent(self):
        """Save the agent state on the server if the endpoint exists."""
        if not self.endpoint_exists("/save_agent"):
            logger.info("save_agent endpoint not available, skipping...")
            return {"success": True, "data": {"message": "save_agent not supported by server"}}
        
        try:
            url = f"{self.base_url}/save_agent"
            response = self.session.post(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Failed to save agent: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def load_agent(self):
        """Load the agent state on the server if the endpoint exists."""
        if not self.endpoint_exists("/load_agent"):
            logger.info("load_agent endpoint not available, skipping...")
            return {"success": True, "data": {"message": "load_agent not supported by server"}}
        
        try:
            url = f"{self.base_url}/load_agent"
            response = self.session.post(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Failed to load agent: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def health_check(self):
        """Check server health."""
        url = f"{self.base_url}/health"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()


def parse_command_line_arguments():
    """Parse and return command line arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        '--server_url',
        type=str,
        default='http://localhost:8000',
        help='Base URL of the MemAgent server'
    )
    parser.add_argument(
        '--dataset', 
        type=str, 
        required=True,
        help='Dataset name (used for output path generation)'
    )
    parser.add_argument(
        '--max_test_queries', 
        type=int, 
        default=0,
        help='Limit maximum test queries (0 = no limit)'
    )
    parser.add_argument(
        '--force', 
        action='store_true', 
        default=False,
        help='Force re-run even if results already exist'
    )
    return parser.parse_args()


def should_skip_processed_context(force_rerun, current_context_index, last_processed_context_id):
    """Determine if we should skip a context that has already been processed."""
    return not force_rerun and current_context_index < last_processed_context_id


def should_skip_processed_query(global_query_index, last_processed_query_id):
    """Determine if we should skip a query that has already been processed."""
    return global_query_index < last_processed_query_id


def should_stop_due_to_query_limit(max_test_queries_ablation, global_query_index):
    """Determine if we should stop processing due to reaching the query limit."""
    return max_test_queries_ablation > 0 and global_query_index >= max_test_queries_ablation


def save_results_to_file(output_path, agent_config, dataset_config, results, metrics, time_cost_list, start_time):
    """Save current results to the output file."""
    # Calculate averaged metrics for logging
    averaged_metrics = {
        key: np.mean(values) * (100 if "_len" not in key else 1) 
        for key, values in metrics.items()
    }
    
    # Log current metrics
    for key, value in averaged_metrics.items():
        logger.info(f"{key}: {value:.02f}")
    
    # Prepare output data structure
    time_cost_list.append(time.time() - start_time)
    output_data = {
        "agent_config": agent_config,
        "dataset_config": dataset_config,
        "data": results,
        "metrics": metrics,
        "time_cost": time_cost_list,
        "averaged_metrics": averaged_metrics,
    }
    
    # Write to file
    with open(output_path, "w") as file:
        json.dump(output_data, file, indent=4)
    logger.info(f"Results saved at {output_path}")


def memorize_context_chunks(client, context_chunks, current_context_index, total_contexts_count):
    """Handle the memorization process for context chunks via API."""
    print("\n\n Agent Memorizing via API...\n\n")
    
    progress_description = f"Processing experiments {current_context_index + 1}/{total_contexts_count}"
    
    for chunk in tqdm(context_chunks, total=len(context_chunks), desc=progress_description):
        try:
            response = client.write_message(chunk, context_id=current_context_index)
            if not response.get('success', False):
                logger.error(f"Failed to memorize chunk: {response.get('error', 'Unknown error')}")
        except Exception as e:
            logger.error(f"Error memorizing chunk: {str(e)}")
            raise


def initialize_and_memorize_agent_via_api(client, context_chunks, current_context_index, total_contexts_count):
    """Initialize agent via API and handle memorization if needed."""
    
    # Initialize the agent via API (just clears memories for new context)
    try:
        response = client.initialize_agent(context_id=current_context_index)
        logger.info(f"Agent initialized: {response['data']['message']}")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {str(e)}")
        raise
    
    # Memorize the context chunks
    try:
        memorize_context_chunks(client, context_chunks, current_context_index, total_contexts_count)
        # Save agent state after memorization (if endpoint exists)
        save_response = client.save_agent()
        if save_response.get('success', False):
            logger.info(f"Agent saved: {save_response['data']['message']}")
    except Exception as e:
        logger.warning(f"Memorization or saving failed: {str(e)}")


def process_single_query(client, query, answer, dataset_config, metrics, results, 
                        global_query_index, current_context_index):
    """Process a single query via API and update metrics and results."""
    try:
        # Send query to agent via API
        response = client.query_message(
            query, 
            query_id=global_query_index, 
            context_id=current_context_index
        )
        
        if not response.get('success', False):
            raise Exception(f"Query failed: {response.get('error', 'Unknown error')}")
        
        agent_output = response['data']['result']
        
        # Calculate metrics and update results
        updated_metrics, updated_results = metrics_summarization(
            agent_output, query, answer, dataset_config, metrics, results, global_query_index
        )
        
        return updated_metrics, updated_results
        
    except Exception as e:
        logger.error(f"Error processing query {global_query_index}: {str(e)}")
        # Return dummy output to avoid breaking the pipeline
        dummy_output = {
            "output": f"Error: {str(e)}",
            "input_len": 0,
            "output_len": 0,
            "memory_construction_time": 0,
            "query_time_len": 0,
        }
        return metrics_summarization(
            dummy_output, query, answer, dataset_config, metrics, results, global_query_index
        )


def process_queries_for_context(client, query_answer_pairs, dataset_config, metrics, results,
                               global_query_index, current_context_index, 
                               last_processed_query_id, max_test_queries,
                               output_path, time_cost_list, start_time):
    """Process all queries for a given context via API."""
    print(f"\n!!!!!Processing {len(query_answer_pairs)} queries for context {current_context_index} via API!!!!!\n")
    for query, answer in tqdm(query_answer_pairs, total=len(query_answer_pairs)):
        # Skip queries that have already been processed
        if should_skip_processed_query(global_query_index, last_processed_query_id):
            logger.info(f"!!!!!Query {global_query_index} already processed, skipping...\n")
            global_query_index += 1
            continue
        
        # Process the current query
        metrics, results = process_single_query(
            client, query, answer, dataset_config, metrics, results,
            global_query_index, current_context_index
        )
        global_query_index += 1
        
        # Save results after each query (freq = 1)
        save_results_to_file(
            output_path, {}, dataset_config, results, 
            metrics, time_cost_list, start_time
        )
        
        # Check if we've reached the query limit
        if should_stop_due_to_query_limit(max_test_queries, global_query_index):
            break
    
    return metrics, results, global_query_index


def main():
    """Main function to run the memory agent benchmark evaluation via API."""
    # Parse command line arguments
    args = parse_command_line_arguments()
    
    # Initialize API client
    client = AgentClient(args.server_url)
    
    # Check server health
    try:
        health = client.health_check()
        logger.info(f"Server health: {health}")
    except Exception as e:
        logger.error(f"Cannot connect to server at {args.server_url}: {str(e)}")
        logger.error("Please make sure the server is running with appropriate configs")
        return
    
    # Create simplified output directory and file path
    output_dir = f"./results/{args.dataset}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "client_results.json")
    
    # Create simplified dataset config for basic functionality
    dataset_config = {
        'dataset': args.dataset,
        'max_test_queries': args.max_test_queries if args.max_test_queries > 0 else 1000000
    }
    
    start_time = time.time()
    time_cost_list = []
    metrics = defaultdict(list)
    results = []
    
    # For this simplified client, we'll demonstrate with a simple test
    # In a real scenario, you'd either:
    # 1. Load data from the server
    # 2. Have a separate data loading mechanism
    # 3. Use the original main.py approach
    
    logger.info("Running simplified client demo - using test data")
    
    # Example test data (replace with actual data loading logic)
    test_context = [
        "The EventQA dataset contains questions about events in narrative texts.",
        "The questions test comprehension of temporal relationships and event sequences.",
        "GPT-4o-mini is used as the language model for this evaluation."
    ]
    
    test_queries = [
        ("What dataset is being used for evaluation?", "EventQA dataset"),
        ("What model is being evaluated?", "GPT-4o-mini"),
        ("What type of questions does the dataset contain?", "Questions about events in narrative texts")
    ]
    
    all_context_chunks = [test_context]
    all_query_answer_pairs = [test_queries]
    
    last_processed_context_id = 0
    last_processed_query_id = 0
    
    # Start evaluation loop - process each context and its associated queries
    global_query_index = 0  # Tracks total queries processed across all contexts
    total_contexts = len(all_context_chunks)
    
    for current_context_index, (context_chunks, query_answer_pairs) in enumerate(
        tqdm(zip(all_context_chunks, all_query_answer_pairs), total=total_contexts)
    ):
        # Skip contexts that have already been fully processed
        if should_skip_processed_context(args.force, current_context_index, last_processed_context_id):
            logger.info(f"\n\n!!!!!Experiment {current_context_index} already finished, skipping...\n")
            global_query_index += len(query_answer_pairs)
            continue
        
        # Initialize and memorize agent for the current context via API
        initialize_and_memorize_agent_via_api(
            client=client,
            context_chunks=context_chunks,
            current_context_index=current_context_index,
            total_contexts_count=total_contexts
        )
        
        # Process all queries for this context via API
        metrics, results, global_query_index = process_queries_for_context(
            client, query_answer_pairs, dataset_config, metrics, results,
            global_query_index, current_context_index, last_processed_query_id,
            args.max_test_queries, output_path, 
            time_cost_list, start_time
        )
        
        # Break early if we've reached the query limit
        if should_stop_due_to_query_limit(args.max_test_queries, global_query_index):
            break
    
    # Log completion
    end_time = time.time()
    logger.info(f"Total time taken: {end_time - start_time}")


if __name__ == '__main__':
    main() 
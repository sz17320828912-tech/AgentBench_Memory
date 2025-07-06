#!/bin/bash

# MemAgent Server Startup Script

echo "Starting MemAgent Server..."

# Default values
HOST="0.0.0.0"
PORT="8000"
RELOAD=""
AGENT_CONFIG=""
DATASET_CONFIG=""
CHUNK_SIZE_ABLATION=0
MAX_TEST_QUERIES_ABLATION=0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --reload)
            RELOAD="--reload"
            shift
            ;;
        --agent_config)
            AGENT_CONFIG="$2"
            shift 2
            ;;
        --dataset_config)
            DATASET_CONFIG="$2"
            shift 2
            ;;
        --chunk_size_ablation)
            CHUNK_SIZE_ABLATION="$2"
            shift 2
            ;;
        --max_test_queries_ablation)
            MAX_TEST_QUERIES_ABLATION="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --agent_config CONFIG --dataset_config CONFIG [OPTIONS]"
            echo "Required:"
            echo "  --agent_config CONFIG     Path to agent configuration file"
            echo "  --dataset_config CONFIG   Path to dataset configuration file"
            echo "Optional:"
            echo "  --host HOST               Host to bind the server to (default: 0.0.0.0)"
            echo "  --port PORT               Port to bind the server to (default: 8000)"
            echo "  --chunk_size_ablation N   Override chunk size for ablation studies (default: 0)"
            echo "  --max_test_queries_ablation N  Limit maximum test queries (default: 0)"
            echo "  --reload                  Enable auto-reload for development"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Check required parameters
if [[ -z "$AGENT_CONFIG" ]]; then
    echo "Error: --agent_config is required"
    echo "Use --help for usage information"
    exit 1
fi

if [[ -z "$DATASET_CONFIG" ]]; then
    echo "Error: --dataset_config is required"
    echo "Use --help for usage information"
    exit 1
fi

# Start the server
echo "Server will be available at http://${HOST}:${PORT}"
echo "API documentation will be available at http://${HOST}:${PORT}/docs"
echo "Agent config: $AGENT_CONFIG"
echo "Dataset config: $DATASET_CONFIG"
echo ""

python server.py \
    --host $HOST \
    --port $PORT \
    --agent_config "$AGENT_CONFIG" \
    --dataset_config "$DATASET_CONFIG" \
    --chunk_size_ablation $CHUNK_SIZE_ABLATION \
    --max_test_queries_ablation $MAX_TEST_QUERIES_ABLATION \
    $RELOAD 
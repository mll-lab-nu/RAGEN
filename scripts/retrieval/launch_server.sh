#!/bin/bash
#
# Launch script for the dense-only retrieval server.
#
# Usage:
#     bash launch_server.sh [data_dir] [port]
#

# Default values
DATA_DIR=${1:-"./search_data/prebuilt_indices"}
PORT=${2:-8000}

echo "Starting dense-only retrieval server..."
echo "Data directory: $DATA_DIR"
echo "Port: $PORT"

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory '$DATA_DIR' not found!"
    echo "Please run download_search_data.py first:"
    echo "  python examples/search/download_search_data.py"
    exit 1
fi

# Check for required files
required_files=("corpus.json" "e5_Flat.index")
for file in "${required_files[@]}"; do
    if [ ! -f "$DATA_DIR/$file" ]; then
        echo "Error: $file not found in $DATA_DIR"
        echo "Please run download_search_data.py to setup data"
        exit 1
    fi
done

# Start server
echo "Launching dense-only server..."
python retrieval/server.py --data_dir "$DATA_DIR" --port "$PORT" --host 0.0.0.0

echo "Server stopped." 
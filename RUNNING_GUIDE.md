# Running Guide After Code Updates

This guide explains how to run the code after the recent changes that replaced hardcoded paths with environment variables.

## Quick Start (Using Defaults)

If you're okay with the default paths, you can run the code directly:

```bash
# For search environment training
bash train_search_multi_dataset.sh
```

The defaults are:
- **Checkpoints**: `~/RAGEN/checkpoints`
- **Cache**: `~/.cache/huggingface`
- **Temporary files**: `/tmp` (or `$TMPDIR` if set)
- **Data directory**: `~/RAGEN/data`

## Customizing Paths (Optional)

If you want to use custom paths, set these environment variables **before** running the script:

```bash
# Set custom paths (optional)
export RAGEN_CHECKPOINTS_DIR="/path/to/your/checkpoints"
export RAGEN_CACHE_DIR="/path/to/your/cache"
export RAGEN_TMP_DIR="/path/to/your/tmp"
export RAGEN_DATA_DIR="/path/to/your/data"

# Then run the script
bash train_search_multi_dataset.sh
```

## Environment Variables Reference

### For Training Scripts (`train_search_multi_dataset.sh`)

| Variable | Default | Description |
|----------|---------|-------------|
| `RAGEN_CHECKPOINTS_DIR` | `~/RAGEN/checkpoints` | Directory for saving model checkpoints |
| `RAGEN_CACHE_DIR` | `~/.cache/huggingface` | Directory for HuggingFace model/dataset cache |
| `RAGEN_TMP_DIR` | `/tmp` or `$TMPDIR` | Directory for temporary files |

### For Search Environment Config

| Variable | Default | Description |
|----------|---------|-------------|
| `RAGEN_DATA_DIR` | `~/RAGEN/data` | Base directory for datasets. The search dataset path will be `$RAGEN_DATA_DIR/search/datasets/search_data.jsonl` |

## Example: Setting Up Custom Paths

```bash
# Example: Using a shared filesystem
export RAGEN_CHECKPOINTS_DIR="/share/nikola/js3673/checkpoints"
export RAGEN_CACHE_DIR="/share/nikola/js3673/cache"
export RAGEN_TMP_DIR="/share/nikola/js3673/tmp"
export RAGEN_DATA_DIR="/share/nikola/js3673/data"

# Run training
bash train_search_multi_dataset.sh
```

## Running Individual Training Commands

You can also run training directly with `train.py`:

```bash
# Basic training (uses defaults)
python train.py --config-name base

# With custom paths via environment variables
export RAGEN_DATA_DIR="/custom/data/path"
python train.py --config-name base
```

## Important Notes

1. **Search Server**: The training script checks if a search server is running at `http://127.0.0.1:8000`. Make sure to start it first:
   ```bash
   cd Search-R1 && bash retrieval_launch.sh
   ```

2. **Directory Creation**: The code will create directories if they don't exist, but make sure you have write permissions.

3. **Backward Compatibility**: If you were using the old hardcoded paths (`/home/js3673/RAGEN/...`), you can set them as environment variables:
   ```bash
   export RAGEN_CHECKPOINTS_DIR="/home/js3673/RAGEN/checkpoints"
   export RAGEN_CACHE_DIR="/share/nikola/js3673/cache"
   export RAGEN_TMP_DIR="/share/nikola/js3673/tmp"
   export RAGEN_DATA_DIR="/home/js3673/RAGEN/data"
   ```

## Checking if the Retriever Server is Running

Before running training, you need to ensure the search/retriever server is running. Here are several ways to check:

### Method 1: Using curl (Recommended)

```bash
# Check if server responds (default URL: http://127.0.0.1:8000)
curl -s --connect-timeout 2 http://127.0.0.1:8000/retrieve

# Or check the root endpoint
curl -s --connect-timeout 2 http://127.0.0.1:8000/

# With verbose output to see response
curl -v http://127.0.0.1:8000/retrieve
```

**Expected behavior:**
- If server is running: You'll get a response (may be an error about missing parameters, but that's OK - it means the server is up)
- If server is not running: `Connection refused` or timeout error

### Method 2: Using wget

```bash
wget -q --spider --timeout=2 http://127.0.0.1:8000/retrieve && echo "Server is running" || echo "Server is not running"
```

### Method 3: Using Python

```bash
python3 -c "
import requests
try:
    response = requests.get('http://127.0.0.1:8000/retrieve', timeout=2)
    print(f'✓ Server is running (status: {response.status_code})')
except requests.exceptions.ConnectionError:
    print('✗ Server is not running (connection refused)')
except requests.exceptions.Timeout:
    print('✗ Server is not responding (timeout)')
except Exception as e:
    print(f'✗ Error: {e}')
"
```

### Method 4: Check if the process is running

```bash
# Check for common server process names
ps aux | grep -E "(retrieval|search|uvicorn|gunicorn|fastapi)" | grep -v grep

# Or check if port 8000 is in use
lsof -i :8000
# or
netstat -tuln | grep 8000
# or
ss -tuln | grep 8000
```

### Method 5: Test with a simple query

```bash
# Send a test query to the retrieve endpoint
curl -X POST http://127.0.0.1:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query_list": ["test query"], "topk": 1}' \
  --timeout 5
```

**Expected response:** JSON with search results or an error message (both indicate the server is running)

### Quick Check Script

You can create a simple script to check:

```bash
#!/bin/bash
# check_server.sh
SERVER_URL="${1:-http://127.0.0.1:8000}"

if curl -s --connect-timeout 2 "${SERVER_URL}/retrieve" > /dev/null 2>&1 || \
   curl -s --connect-timeout 2 "${SERVER_URL}/" > /dev/null 2>&1; then
    echo "✓ Search server is running at ${SERVER_URL}"
    exit 0
else
    echo "✗ Search server is NOT running at ${SERVER_URL}"
    echo "  Please start it with: cd Search-R1 && bash retrieval_launch.sh"
    exit 1
fi
```

Then run: `bash check_server.sh`

### Starting the Server

If the server is not running, start it:

```bash
# Navigate to Search-R1 directory and start the server
cd Search-R1
bash retrieval_launch.sh

# Or if you have a different setup:
# python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

### Custom Server URL

If your server is running on a different URL, you can check it:

```bash
# Check custom URL
curl -s --connect-timeout 2 http://your-server:8000/retrieve

# Or set it as an environment variable and check
export RAGEN_SERVER_URL="http://your-server:8000"
curl -s --connect-timeout 2 "${RAGEN_SERVER_URL}/retrieve"
```

## Verifying Your Setup

To check what paths will be used, you can inspect the environment variables:

```bash
# Check what will be used (before running the script)
echo "Checkpoints: ${RAGEN_CHECKPOINTS_DIR:-~/RAGEN/checkpoints}"
echo "Cache: ${RAGEN_CACHE_DIR:-~/.cache/huggingface}"
echo "TMP: ${RAGEN_TMP_DIR:-/tmp}"
echo "Data: ${RAGEN_DATA_DIR:-~/RAGEN/data}"
```

Or add this to your `~/.bashrc` or `~/.zshrc` for persistent settings:

```bash
# Add to ~/.bashrc or ~/.zshrc
export RAGEN_CHECKPOINTS_DIR="/your/preferred/path/checkpoints"
export RAGEN_CACHE_DIR="/your/preferred/path/cache"
export RAGEN_TMP_DIR="/your/preferred/path/tmp"
export RAGEN_DATA_DIR="/your/preferred/path/data"
```


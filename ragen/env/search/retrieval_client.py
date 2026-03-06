"""
Retrieval client for connecting to the dense retrieval server.
Simplified from rllm/examples/search/local_retrieval_tool.py.

The retrieval server (scripts/retrieval/server.py) must be running before use.
Uses `requests` instead of rllm's `httpx` to minimize new dependencies.
"""

import logging
import os
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class RetrievalClient:
    """
    HTTP client for the dense retrieval server (E5 + FAISS).

    Connects to the Flask server at scripts/retrieval/server.py.
    Designed to be fault-tolerant: logs warnings instead of crashing
    when the server is unavailable.
    """

    def __init__(
        self,
        server_url: Optional[str] = None,
        timeout: float = 30.0,
        max_results: int = 10,
        max_total_chars: int = 4000,
    ):
        if not HAS_REQUESTS:
            logger.warning("requests package not installed. RetrievalClient will not work.")

        if server_url is None:
            server_url = os.environ.get("RETRIEVAL_SERVER_URL", "http://127.0.0.1:8000")

        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.max_results = max_results
        self.max_total_chars = max_total_chars
        self.available = False

        self._test_connection()

    def _test_connection(self) -> bool:
        """Test connection to the retrieval server. Warning only, never crashes."""
        if not HAS_REQUESTS:
            return False
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info(f"Connected to retrieval server at {self.server_url}")
                self.available = True
                return True
            else:
                logger.warning(f"Retrieval server returned status {response.status_code}")
                return False
        except Exception as e:
            logger.warning(f"Could not connect to retrieval server at {self.server_url}: {e}")
            return False

    def search(self, query: str, top_k: Optional[int] = None) -> str:
        """
        Execute a search query against the retrieval server.

        Args:
            query: The search query string.
            top_k: Number of results to return (default: self.max_results).

        Returns:
            Formatted search results as a string, or an error message.
        """
        if not HAS_REQUESTS:
            return "Search service unavailable: requests package not installed."

        top_k = top_k or self.max_results

        try:
            payload = {
                "query": query,
                "top_k": min(top_k, 50),
            }

            response = requests.post(
                f"{self.server_url}/retrieve",
                json=payload,
                timeout=self.timeout,
            )

            if not response.ok:
                error_msg = f"Retrieval server error (status {response.status_code})"
                try:
                    error_data = response.json()
                    error_msg += f": {error_data.get('error', 'Unknown error')}"
                except Exception:
                    pass
                return error_msg

            response_data = response.json()
            results = response_data.get("results", [])

            if not results:
                return "No relevant documents found for the query."

            return self._format_results(results)

        except requests.exceptions.Timeout:
            return f"Search request timed out after {self.timeout} seconds."
        except requests.exceptions.ConnectionError:
            return f"Could not connect to retrieval server at {self.server_url}. Is it running?"
        except Exception as e:
            return f"Search error: {str(e)}"

    def _format_results(self, results: List[dict]) -> str:
        """Format search results for LLM consumption. Truncates long documents."""
        formatted = []
        for i, result in enumerate(results[:self.max_results], 1):
            doc_id = result.get("id", f"doc_{i}")
            content = result.get("content", "")
            score = result.get("score", 0.0)

            # Truncate content to 300 chars (same as rllm)
            if len(content) > 800:
                content = content[:800] + "..."

            formatted.append(f"[Document {i}] (ID: {doc_id}, Score: {score:.3f})\n{content}")

        output = "\n\n".join(formatted)

        # Cap total output to max_total_chars (~1k tokens) to prevent context overflow
        if len(output) > self.max_total_chars:
            output = output[:self.max_total_chars] + "..."

        return output


class MockRetrievalClient:
    """
    Mock retrieval client for development/testing without a running server.
    Returns placeholder results so the environment can be tested end-to-end.
    """

    def __init__(self, **kwargs):
        self.available = True
        logger.info("Using MockRetrievalClient (no real retrieval server)")

    def search(self, query: str, top_k: Optional[int] = None) -> str:
        return (
            f"[Document 1] (ID: mock_1, Score: 0.900)\n"
            f"Mock search result for query: '{query}'. "
            f"This is a placeholder document for testing purposes.\n\n"
            f"[Document 2] (ID: mock_2, Score: 0.750)\n"
            f"Another mock document related to: '{query}'. "
            f"Replace with real retrieval server for actual training."
        )

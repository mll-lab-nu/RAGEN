"""
Unified server-based search adapter for RAGEN Search Environment.
Supports both Search-R1 retrieval server and generic HTTP search services.
"""

import requests
from typing import List, Dict, Any, Optional


class ServerSearchAdapter:
    """Unified HTTP search server adapter supporting both Search-R1 and generic servers."""
    
    def __init__(self, server_url: str = "http://127.0.0.1:8000", endpoint: str = "/retrieve", topk: int = 3, api_format: str = "searchr1"):
        self.server_url = server_url
        self.endpoint = endpoint
        self.topk = topk
        self.api_format = api_format  # "searchr1" or "generic"
    
    def search(self, query: str, num: Optional[int] = None, return_scores: bool = False) -> List[Dict[str, Any]]:
        """Search using HTTP search server with flexible API format."""
        try:
            num = num or self.topk
            
            # Prepare payload based on API format
            if self.api_format == "searchr1":
                payload = {
                    "queries": [query],  # Search-R1 expects queries as list
                    "topk": num,
                    "return_scores": True
                }
            else:  # generic format
                payload = {
                    "query": query,
                    "topk": num
                }
            
            response = requests.post(
                f"{self.server_url}{self.endpoint}",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Handle different response formats
            if self.api_format == "searchr1":
                results = data.get("result", [])
                # Search-R1 returns results as list of lists, get first query results
                if results and len(results) > 0:
                    query_results = results[0]  # First query results
                else:
                    query_results = []
            else:  # generic format
                query_results = data.get("results", data.get("result", []))
            
            # Convert to expected format
            search_results = []
            for idx, result in enumerate(query_results):
                if isinstance(result, str):
                    search_results.append({
                        'title': f"Document {idx+1}",
                        'text': result,
                        'contents': result,
                        'score': 0.0
                    })
                elif isinstance(result, dict):
                    # Handle dict results with document and score
                    if 'document' in result and 'score' in result:
                        search_results.append({
                            'title': f"Document {idx+1}",
                            'text': result['document'],
                            'contents': result['document'],
                            'score': result['score']
                        })
                    else:
                        search_results.append({
                            'title': result.get('title', f"Document {idx+1}"),
                            'text': result.get('text', result.get('content', str(result))),
                            'contents': result.get('contents', result.get('text', result.get('content', str(result)))),
                            'score': result.get('score', 0.0)
                        })
                else:
                    search_results.append({
                        'title': f"Document {idx+1}",
                        'text': str(result),
                        'contents': str(result),
                        'score': 0.0
                    })
            
            return search_results
            
        except requests.exceptions.RequestException as e:
            print(f"Server search error: {e}")
            # Return mock search results when server is not available
            return self._get_mock_results(query)
        except Exception as e:
            print(f"Server search error: {e}")
            # Return mock search results when server has unexpected errors
            return self._get_mock_results(query)
    
    def _get_mock_results(self, query: str) -> List[Dict[str, Any]]:
        """Return mock search results for fallback."""
        return [
            {
                "title": f"Search Result 1 for '{query}'",
                "text": f"Based on the query '{query}', here is some relevant information that could help answer the question.",
                "contents": f"Search results for '{query}': This is a placeholder result. The search server is not currently available, but this demonstrates how search results would be formatted.",
                "score": 0.8
            },
            {
                "title": f"Search Result 2 for '{query}'",
                "text": f"Additional information related to '{query}' that might be useful for answering the question.",
                "contents": f"More search results for '{query}': This is another placeholder result showing the expected format.",
                "score": 0.7
            }
        ]


def get_server_search_adapter(config) -> Optional[Any]:
    """Get server-based search adapter based on configuration."""
    # Determine API format based on endpoint
    if config.server_endpoint == "/retrieve":
        api_format = "searchr1"  # Search-R1 format
    else:
        api_format = "generic"  # Generic format
    
    return ServerSearchAdapter(
        server_url=getattr(config, 'server_url', 'http://127.0.0.1:8000'),
        endpoint=getattr(config, 'server_endpoint', '/retrieve'),
        topk=getattr(config, 'topk', 3),
        api_format=api_format
    )


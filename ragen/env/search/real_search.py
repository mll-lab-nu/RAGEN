"""
Real search adapters for RAGEN search environment.
Supports Google Custom Search API and SerpAPI.
"""
import requests
from typing import List, Optional, Dict, Any
from googleapiclient.discovery import build


class GoogleSearchAdapter:
    """Google Custom Search API adapter."""
    
    def __init__(self, api_key: str, cse_id: str, topk: int = 3, snippet_only: bool = False):
        self.api_key = api_key
        self.cse_id = cse_id
        self.topk = topk
        self.snippet_only = snippet_only
        self.service = build("customsearch", "v1", developerKey=api_key)
    
    def search(self, query: str, topk: Optional[int] = None) -> List[Dict[str, Any]]:
        """Perform Google Custom Search."""
        try:
            num = topk or self.topk
            results = self.service.cse().list(
                q=query,
                cx=self.cse_id,
                num=min(num, 10)  # Google API limit
            ).execute()
            
            search_results = []
            for item in results.get('items', [])[:num]:
                result = {
                    'title': item.get('title', ''),
                    'text': item.get('snippet', ''),
                    'url': item.get('link', ''),
                    'contents': f"{item.get('title', '')}\n{item.get('snippet', '')}"
                }
                search_results.append(result)
            
            return search_results
        except Exception as e:
            print(f"Google Search error: {e}")
            return []


class SerpAPISearchAdapter:
    """SerpAPI search adapter."""
    
    def __init__(self, api_key: str, search_url: str = "https://serpapi.com/search", 
                 engine: str = "google", topk: int = 3):
        self.api_key = api_key
        self.search_url = search_url
        self.engine = engine
        self.topk = topk
    
    def search(self, query: str, topk: Optional[int] = None) -> List[Dict[str, Any]]:
        """Perform SerpAPI search."""
        try:
            num = topk or self.topk
            params = {
                'api_key': self.api_key,
                'q': query,
                'engine': self.engine,
                'num': min(num, 10)
            }
            
            response = requests.get(self.search_url, params=params, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            
            search_results = []
            for result in data.get('organic_results', [])[:num]:
                search_result = {
                    'title': result.get('title', ''),
                    'text': result.get('snippet', ''),
                    'url': result.get('link', ''),
                    'contents': f"{result.get('title', '')}\n{result.get('snippet', '')}"
                }
                search_results.append(search_result)
            
            return search_results
        except requests.exceptions.RequestException as e:
            print(f"SerpAPI Search error (network/HTTP): {e}")
            return []
        except (ValueError, KeyError) as e:
            print(f"SerpAPI Search error (parsing): {e}")
            return []
        except Exception as e:
            print(f"SerpAPI Search error (unexpected): {e}")
            return []


class SerperSearchAdapter:
    """Serper API search adapter."""
    
    def __init__(self, api_key: str, topk: int = 3):
        self.api_key = api_key
        self.topk = topk
        self.base_url = "https://google.serper.dev/search"
        self.headers = {
            'Content-Type': 'application/json'
        }
    
    def search(self, query: str, topk: Optional[int] = None) -> List[Dict[str, Any]]:
        """Perform Serper API search."""
        try:
            num = topk or self.topk
            params = {
                'q': query,
                'apiKey': self.api_key,
                'num': min(num, 10)
            }
            
            response = requests.get(self.base_url, params=params, headers=self.headers, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            
            search_results = []
            for result in data.get('organic', [])[:num]:
                search_result = {
                    'title': result.get('title', ''),
                    'text': result.get('snippet', ''),
                    'url': result.get('link', ''),
                    'contents': f"{result.get('title', '')}\n{result.get('snippet', '')}"
                }
                search_results.append(search_result)
            
            return search_results
        except requests.exceptions.RequestException as e:
            print(f"Serper API Search error (network/HTTP): {e}")
            return []
        except (ValueError, KeyError) as e:
            print(f"Serper API Search error (parsing): {e}")
            return []
        except Exception as e:
            print(f"Serper API Search error (unexpected): {e}")
            return []


def get_real_search_adapter(config) -> Optional[Any]:
    """Get the appropriate real search adapter based on configuration."""
    if config.search_type == "google":
        if not config.google_api_key or not config.google_cse_id:
            print("Warning: Google search requires google_api_key and google_cse_id")
            return None
        return GoogleSearchAdapter(
            api_key=config.google_api_key,
            cse_id=config.google_cse_id,
            topk=config.topk,
            snippet_only=config.google_snippet_only
        )
    
    elif config.search_type == "serpapi":
        if not config.serp_api_key:
            print("Warning: SerpAPI search requires serp_api_key")
            return None
        return SerpAPISearchAdapter(
            api_key=config.serp_api_key,
            search_url=config.serp_search_url,
            engine=config.serp_engine,
            topk=config.topk
        )
    
    elif config.search_type == "serper":
        if not config.serper_api_key:
            print("Warning: Serper search requires serper_api_key")
            return None
        return SerperSearchAdapter(
            api_key=config.serper_api_key,
            topk=config.topk
        )
    
    else:
        return None


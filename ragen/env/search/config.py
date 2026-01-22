import os
from ragen.env.base import BaseEnvConfig
from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class SearchEnvConfig(BaseEnvConfig):
    """Configuration for Search environment"""
    # Data configuration - supports Search-R1 datasets
    dataset_path: str = field(default_factory=lambda: os.path.join(
        os.getenv("RAGEN_DATA_DIR", os.path.join(os.path.expanduser("~"), "RAGEN", "data")),
        "search", "datasets", "search_data.jsonl"
    ))
    max_instances: int = field(default=1000)
    data_source: str = field(default="nq")  # nq, triviaqa, popqa, web_questions, hotpotqa, 2wikimultihopqa, musique, bamboogle, strategyqa, eli5
    template_type: str = field(default="base")  # Search-R1 template type
    mode: str = field(default="train")  # train or val/test - determines which data split to load (like Search-R1)
    
    # Environment configuration
    max_steps: int = 6
    topk: int = 3
    max_length_per_doc: int = field(default=200)  # Maximum length per document in search results
    
    # Search configuration
    search_type: str = field(default="server")  # server (Search-R1), google, serpapi, serper
    
    # Server-based search configuration
    server_url: str = field(default="http://127.0.0.1:8000")  # Search server URL
    server_endpoint: str = field(default="/retrieve")  # Server endpoint for generic servers
    
    # Search-R1 server configuration (these are handled by the Search-R1 server)
    # No need for local configuration since Search-R1 server handles everything
    
    # Google Search API configuration
    google_api_key: str = field(default="")  # Google Custom Search API key
    google_cse_id: str = field(default="")  # Google Custom Search Engine ID
    google_snippet_only: bool = field(default=False)  # Return only snippets or full content
    
    # SerpAPI configuration
    serp_api_key: str = field(default="")  # SerpAPI key
    serp_engine: str = field(default="google")  # Search engine (google, bing, etc.)
    serp_search_url: str = field(default="https://serpapi.com/search")  # SerpAPI endpoint
    
    # Serper API configuration
    serper_api_key: str = field(default="")  # Serper API key
    
    # Reward scoring configuration (Search-R1 style)
    reward_type: str = field(default="em")  # "em" for exact match, "rouge" for ROUGE score (for ELI5)
    rouge_type: str = field(default="rougeL")  # ROUGE type: rouge1, rouge2, rougeL, rougeLsum
    structure_format_score: float = field(default=0.2)  # Score for correct format structure
    final_format_score: float = field(default=0.1)      # Score for final answer format  
    retrieval_score: float = field(default=0.0)         # Score for successful retrieval (Search-R1 uses 0)
    score: float = field(default=1.0)                   # Score for correct answer


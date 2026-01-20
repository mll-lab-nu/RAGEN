import re
import json
import random
from typing import Optional, Tuple, Any, Dict, List
from ragen.env.base import BaseLanguageBasedEnv
from ragen.env.search.config import SearchEnvConfig
from ragen.env.search.utils.reward_score.qa_em import em_check, normalize_answer
from ragen.env.search.utils.reward_score.qa_rouge import compute_score_rouge
from ragen.utils import all_seed
import datasets


class SearchEnv(BaseLanguageBasedEnv):
    """Multi-turn search environment following Search-R1's pattern.
    
    Actions: LLM generates <search>query</search> or <answer>response</answer>
    Environment: Executes search calls and appends results as <information> blocks
    """

    def __init__(self, config: Optional[SearchEnvConfig] = None) -> None:
        super().__init__()
        self.config = config or SearchEnvConfig()
        self._obs: Optional[str] = None
        self._steps = 0
        self._history: List[str] = []
        self._question: str = ""
        self._ground_truth: Optional[Dict] = None
        
        # Load data from configuration
        self.data = self._load_data()
        self.current_data_idx = None

        # Initialize search (local or real search)
        self._search = self._init_search()

    def _init_search(self):
        """Initialize search (server or real search) based on configuration."""
        if self.config.search_type == "server":
            return self._init_server_search()
        else:
            return self._init_real_search()
    
    
    def _init_real_search(self):
        """Initialize real search (Google, SerpAPI, etc.)."""
        try:
            from .real_search import get_real_search_adapter
            return get_real_search_adapter(self.config)
        except Exception as e:
            print(f"Error: Could not initialize real search: {e}")
            print("Search actions will return empty results.")
            return None
    
    def _init_server_search(self):
        """Initialize server-based search (Search-R1 server, etc.)."""
        try:
            from .server_search import get_server_search_adapter
            return get_server_search_adapter(self.config)
        except Exception as e:
            print(f"Error: Could not initialize server search: {e}")
            print("Search actions will return empty results.")
            return None

    def _load_data(self) -> List[Dict]:
        """Load search data from configuration - supports both local files and HuggingFace datasets.
        
        Matches Search-R1's approach: uses train split for training, test split for validation/testing.
        This ensures proper train/test separation to avoid data leakage.
        """
        try:
            # Get mode from config (set by es_manager based on train/val mode)
            mode = getattr(self.config, 'mode', 'train')
            
            # Try to load from HuggingFace dataset first (like Search-R1)
            if self.config.data_source in ['nq', 'triviaqa', 'popqa', 'web_questions', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle', 'eli5']:
                # Load dataset - will use cache if available, download if not
                dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', self.config.data_source)
                
                # Use train split for training, dev split for validation (consistent with FlashRAG datasets)
                if mode == 'train':
                    # For training: use train split
                    if 'train' in dataset:
                        data_split = dataset['train']
                    else:
                        # Fallback if train split doesn't exist
                        print(f"Warning: 'train' split not found for {self.config.data_source}, using first available split")
                        data_split = list(dataset.values())[0]
                else:
                    # For validation/testing: prefer dev split, then test split, then train as fallback
                    if 'dev' in dataset:
                        data_split = dataset['dev']
                    elif 'test' in dataset:
                        data_split = dataset['test']
                    else:
                        print(f"Warning: 'dev' or 'test' split not found for {self.config.data_source}, using train split")
                        data_split = dataset.get('train', list(dataset.values())[0])
                
                # Convert to our format
                data = []
                # For validation/evaluation, use the full dev set; for training, limit by max_instances
                max_instances_to_use = None if mode != 'train' else self.config.max_instances
                
                for i, example in enumerate(data_split):
                    if max_instances_to_use is not None and i >= max_instances_to_use:
                        break
                    
                    # Format question
                    question = example['question'].strip()
                    if question[-1] != '?':
                        question += '?'
                    
                    data.append({
                        "question": question,
                        "ground_truth": {
                            "target": example['golden_answers']
                        },
                        "data_source": self.config.data_source
                    })
                return data
                
            else:
                # Fallback to local JSONL file
                with open(self.config.dataset_path, 'r') as f:
                    data = []
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))
                    return data[:self.config.max_instances]
                    
        except Exception as e:
            raise RuntimeError(
                f"Failed to load dataset '{self.config.data_source}': {e}\n"
                f"Please check your configuration and ensure the dataset is available."
            ) from e

    def reset(self, seed=None, data_idx=None, **kwargs) -> Any:
        self._steps = 0
        self._history = []
        self._pending_search_results = None  # Store search results for next turn
        
        # Select a question from the dataset
        if data_idx is not None:
            # Use provided index (for sampling without replacement)
            self.current_data_idx = data_idx % len(self.data)  # Wrap around if needed
        else:
            # Fallback to random selection (for backward compatibility)
            with all_seed(seed):
                self.current_data_idx = random.randint(0, len(self.data) - 1)
        
        data_item = self.data[self.current_data_idx]
        self._question = data_item["question"]
        self._ground_truth = data_item["ground_truth"]
        
        # Build observation with Search-R1 style instruction
        self._obs = f"Question: {self._question}\n\n"
        
        # Add pending search results if any (from previous search action)
        if self._pending_search_results:
            self._obs += f"<information>{self._pending_search_results}</information>\n\n"
            self._pending_search_results = None  # Clear after adding
        
        return self.render()

    def step(self, action: str) -> Tuple[Any, float, bool, Dict]:
        self._steps += 1
        info: Dict = {}
        action = (action or "").strip()
        
        # Rebuild observation from scratch (Search-R1 style)
        # For subsequent turns, only include the information block (not the question)
        # The question is already in the prefix/instruction managed by context manager
        if self._pending_search_results:
            # Only include the information block, not the question (matches Search-R1)
            # The question appears once in the prefix, not repeated in each turn's state
            self._obs = f"<information>{self._pending_search_results}</information>\n\n"
            self._pending_search_results = None  # Clear after adding
        else:
            # First turn or no pending results: include question for initial state
            self._obs = f"Question: {self._question}\n\n"
        
        # Parse action using Search-R1 pattern
        action_type, content = self._parse_action(action)
        is_valid = action_type is not None
        
        if action_type == "search":
            # Execute search and get results (Search-R1 style)
            results = self._execute_search(content)
            formatted_results = self._format_search_results(results)
            
            # Search-R1 pattern: return ONLY the <information> block as observation from step()
            # This matches Search-R1's execute_predictions which returns just the info block
            # The question and context are already in the history managed by context manager
            obs = f'\n\n<information>{formatted_results.strip()}</information>\n\n'
            
            # Rebuild self._obs for the NEXT turn (when render() is called after this step)
            # Only include the information block, not the question (to avoid prompt length issues)
            # The question appears once in the prefix, not repeated in each turn's state
            self._obs = f"<information>{formatted_results.strip()}</information>\n\n"
            
            # Clear pending results since we've already included them in self._obs
            self._pending_search_results = None
            self._history.append(f"search: {content}")
            
            # Format reward is given when action has thinking tag followed by search/answer tag
            # Check if action has proper format: <think>...</think> followed by <search>...</search>
            has_thinking_tag = '<think>' in action and '</think>' in action
            has_both_tags = '<search>' in action and '<answer>' in action
            
            if has_both_tags or not has_thinking_tag:
                # Multiple actions or missing thinking tag: no format reward
                reward = 0.0
            else:
                # Single action with proper format (thinking + search): give format reward
                reward = self.config.final_format_score
            
            done = False
            info.update({
                "action_is_valid": is_valid,
                "action_is_effective": True,
                "is_search": True,
                "success": 0.0
            })
            
            return obs, reward, done, info
        elif action_type == "answer":
            # Final answer - end episode and compute reward
            # Use different reward functions based on dataset type
            accuracy = 0.0
            format_component = 0.0
            if self.config.reward_type == "rouge":
                # ROUGE-based reward for ELI5 (long-form answers)
                # Format reward is only given when action has thinking tag followed by answer/search tag
                # Check if action has proper format: <think>...</think> followed by <answer>...</answer>
                has_thinking_tag = '<think>' in action and '</think>' in action
                has_both_tags = '<search>' in action and '<answer>' in action
                
                if has_both_tags or not has_thinking_tag:
                    # Multiple actions or missing thinking tag: no format reward
                    reward = 0.0
                    rouge_metrics = {"total_reward": 0.0, "rouge_score": 0.0, "format_reward": 0.0}
                else:
                    # Single action with proper format (thinking + answer): compute ROUGE reward
                    # Format reward is built into compute_score_rouge (uses final_format_score)
                    solution_str = action
                    rouge_metrics = compute_score_rouge(
                        solution_str=solution_str,
                        ground_truth=self._ground_truth,
                        method=self.config.rouge_type,
                        format_score=self.config.final_format_score,  # Same as search actions
                        score=self.config.score,
                        return_dict=True
                    )
                    reward = rouge_metrics["total_reward"]
            else:
                # Exact match reward for other datasets (Search-R1 style)
                # Normalize the answer for comparison
                normalized_answer = normalize_answer(content)
                normalized_targets = [normalize_answer(target) for target in self._ground_truth['target']]
                
                # Check exact match
                if em_check(normalized_answer, normalized_targets):
                    reward = self.config.score  # 1.0 for correct answer
                    accuracy = 1.0
                else:
                    # Check if answer format is reasonable (not empty, not just punctuation)
                    if content.strip() and len(content.strip()) > 1:
                        reward = self.config.final_format_score  # 0.1 for reasonable format
                        format_component = self.config.final_format_score
                    else:
                        reward = 0.0  # No reward for empty/invalid answers
            
            # Append to history after computing reward
            self._history.append(f"answer: {content}")
            
            done = True
            success_flag = float(reward > 0) if self.config.reward_type == "rouge" else accuracy
            info_dict = {
                "action_is_valid": is_valid,
                "action_is_effective": True,
                "is_search": False,
                "success": success_flag
            }
            # Add separate metrics for ROUGE-based rewards (rouge, format_reward, total_reward)
            if self.config.reward_type == "rouge":
                info_dict.update({
                    "rouge": rouge_metrics["rouge_score"],  # ROUGE score (actual correctness)
                    "format_reward": rouge_metrics["format_reward"],  # Format reward component
                    "total_reward": rouge_metrics["total_reward"]  # Total reward (for backward compatibility)
                })
            else:
                info_dict.update({
                    "accuracy": accuracy,
                    "format_reward": format_component,
                    "total_reward": reward
                })
            info.update(info_dict)
        else:
            # Invalid action format - provide specific feedback
            if not action.strip():
                feedback = "\n\n❌ Error: Empty action received. Please provide a valid action.\n"
            elif '<search>' in action and '</search>' not in action:
                feedback = "\n\n❌ Error: Malformed search tag. Please use complete tags: <search>your query</search>\n"
            elif '<answer>' in action and '</answer>' not in action:
                feedback = "\n\n❌ Error: Malformed answer tag. Please use complete tags: <answer>your response</answer>\n"
            elif '<search>' in action and '</search>' in action:
                # Both tags present but parsing failed - check content
                search_start = action.find('<search>')
                search_end = action.find('</search>', search_start)
                content = action[search_start + 8:search_end].strip() if search_start != -1 and search_end != -1 else ""
                if not content or len(content) < 3:
                    feedback = "\n\n❌ Error: Search query is too short or empty. Please provide a meaningful search query.\n"
                else:
                    # Format looks correct but parsing failed - likely due to nested tags or other issues
                    feedback = "\n\n❌ Error: Invalid action format. Please use: <search>your search query</search> (avoid nested tags)\n"
            elif '<answer>' in action and '</answer>' in action:
                # Both tags present but parsing failed - check content
                answer_start = action.find('<answer>')
                answer_end = action.find('</answer>', answer_start)
                content = action[answer_start + 8:answer_end].strip() if answer_start != -1 and answer_end != -1 else ""
                if not content or len(content) < 3:
                    feedback = "\n\n❌ Error: Answer is too short or empty. Please provide a meaningful answer.\n"
                else:
                    # Format looks correct but parsing failed - likely due to nested tags or other issues
                    feedback = "\n\n❌ Error: Invalid action format. Please use: <answer>your answer</answer> (avoid nested tags)\n"
            else:
                feedback = "\n\n❌ Error: Invalid action format. Please use one of:\n- <search>your query</search> to search for information\n- <answer>your response</answer> to provide your final answer\n"
            
            self._obs += feedback
            reward = 0.0
            done = False
            info.update({
                "action_is_valid": False,
                "action_is_effective": False,
                "is_search": False,
                "success": 0.0
            })

        # Check max steps
        if self._steps >= self.config.max_steps:
            done = True
            if not info.get("success", 0):
                reward = 0.0
                # Add timeout message to observation
                self._obs += f"\n\n⏰ Maximum steps ({self.config.max_steps}) reached. Episode ended without success."

        # For answer and invalid actions, return the full observation
        # For search actions, we already returned early with just the information block
        return self.render(), reward, done, info

    def render(self, mode: str = 'text') -> Any:
        return self._obs or ""

    def close(self):
        pass

    def _parse_action(self, action: str) -> Tuple[Optional[str], str]:
        """Parse action supporting <think>reasoning</think><search>query</search> or <think>reasoning</think><answer>response</answer>
        
        Priority: answer > search (if both are present, prefer answer since it ends the episode)
        """
        if not action or not action.strip():
            return None, ""
        
        # Clean up the action string
        action = action.strip()
        
        # First try to find complete, well-formed actions using regex
        pattern = r'<(search|answer)>(.*?)</\1>'
        matches = re.findall(pattern, action, re.DOTALL)
        
        if matches:
            # Priority: prefer 'answer' over 'search' if both are present (answer ends the episode)
            action_type = None
            content = ""
            
            # Check for answer first (preferred since it ends the episode)
            for match_type, match_content in matches:
                if match_type == "answer":
                    action_type = "answer"
                    content = match_content.strip()
                    break
            
            # If no answer found, use the first search
            if action_type is None:
                for match_type, match_content in matches:
                    if match_type == "search":
                        action_type = "search"
                        content = match_content.strip()
                        break
            
            # Validate content is not empty and not just malformed tags
            if action_type and content and len(content) > 0:
                # Remove any nested tags from content
                content = self._clean_content(content)
                if content:  # Only return if we have valid content
                    return action_type, content
        
        # If no complete actions found, try to extract from partial/malformed actions
        # Priority: answer > search (answer ends the episode)
        # Look for answer actions first
        if '<answer>' in action:
            answer_start = action.find('<answer>')
            if answer_start != -1:
                # Find the next </answer> tag
                answer_end = action.find('</answer>', answer_start)
                if answer_end != -1:
                    content = action[answer_start + 8:answer_end].strip()
                    content = self._clean_content(content)
                    if content:  # Only return if we have valid content
                        return "answer", content
        
        # Look for search actions
        if '<search>' in action:
            search_start = action.find('<search>')
            if search_start != -1:
                # Find the next </search> tag
                search_end = action.find('</search>', search_start)
                if search_end != -1:
                    content = action[search_start + 8:search_end].strip()
                    content = self._clean_content(content)
                    if content:  # Only return if we have valid content
                        return "search", content
        
        return None, ""
    
    def _clean_content(self, content: str) -> str:
        """Clean content by removing nested tags"""
        if not content:
            return ""
        
        # Remove any nested search/answer tags (in case model generates nested tags)
        content = re.sub(r'<search>.*?</search>', '', content, flags=re.DOTALL)
        content = re.sub(r'<answer>.*?</answer>', '', content, flags=re.DOTALL)
        
        # Clean up whitespace
        content = content.strip()
        
        return content

    def _execute_search(self, query: str) -> List[Dict]:
        """Execute search using Search-R1 retriever or real search APIs"""
        if not query or not query.strip():
            return []
        
        # If no search adapter is configured, return mock results
        if not self._search:
            return [
                {
                    "title": "Search Not Configured",
                    "text": f"Search for '{query}' would be executed here. Please configure search_type and API keys in your environment configuration.",
                    "contents": f"Mock search result for query: {query}"
                }
            ]
        
        try:
            # Execute search based on adapter type
            if hasattr(self._search, 'search'):
                # Search-R1 style retriever
                results = self._search.search(query, num=self.config.topk)
            elif hasattr(self._search, 'search_query'):
                # Real search API adapter
                results = self._search.search_query(query, num_results=self.config.topk)
            else:
                # Fallback
                results = []
            
            # Ensure results are in the expected format
            if not isinstance(results, list):
                results = []
            
            # If no results, provide a helpful message
            if not results:
                return [
                    {
                        "title": "No Results Found",
                        "text": f"No relevant information found for query: '{query}'. Try rephrasing your search terms.",
                        "contents": f"No results for: {query}"
                    }
                ]
            
            return results
            
        except Exception as e:
            print(f"Search error: {e}")
            return [
                {
                    "title": "Search Error",
                    "text": f"Search failed for query: '{query}'. Error: {str(e)}",
                    "contents": f"Search error for: {query}"
                }
            ]

    def _format_search_results(self, results: List[Dict]) -> str:
        """Format search results following Search-R1's format with length limits"""
        if not results:
            return "No relevant information found."
        
        formatted = ""
        max_length_per_doc = self.config.max_length_per_doc
        
        for idx, doc in enumerate(results):
            if isinstance(doc, dict):
                title = doc.get('title', '')
                text = doc.get('text', '')
                contents = doc.get('contents', text)
                
                # Extract title and text from contents if needed
                if contents and not title:
                    lines = contents.split('\n')
                    title = lines[0] if lines else ''
                    text = '\n'.join(lines[1:]) if len(lines) > 1 else contents
                
                # Limit text length to prevent very long prompts
                if len(text) > max_length_per_doc:
                    text = text[:max_length_per_doc] + "..."
                
                formatted += f"Doc {idx+1}(Title: {title}) {text}\n"
            else:
                doc_str = str(doc)
                if len(doc_str) > max_length_per_doc:
                    doc_str = doc_str[:max_length_per_doc] + "..."
                formatted += f"Doc {idx+1}: {doc_str}\n"
        
        return formatted.strip()


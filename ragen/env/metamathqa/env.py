import gym
from gym import spaces
import numpy as np
from datasets import load_dataset
import re
import random
from ragen.env.base import BaseLanguageBasedEnv
from ragen.utils import all_seed
from .config import MetaMathQAEnvConfig
class MetaMathQAEnv(BaseLanguageBasedEnv):
    def __init__(self, config: MetaMathQAEnvConfig):
        super(MetaMathQAEnv, self).__init__()
        
        self.config = config
        self.dataset = load_dataset(path=self.config.dataset_path, cache_dir=self.config.cache_dir)
        self.current_question_idx = None
        self.current_question = None
        self.correct_answer = None
        self.step_num = None
        self.render_cache = None
        
        
    def _extract_answer(self, response):
        match = re.search(r"The answer is: (.*?)$", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
        
    def reset(self,seed=None, mode=None):
        dataset = self.dataset[self.config.split]
        with all_seed(seed):
            self.current_question_idx = random.randint(0, len(dataset) - 1)
        question_data = dataset[self.current_question_idx]
        self.current_question = question_data['query']
        self.correct_answer = self._extract_answer(question_data['response'])
        self.step_num = 0
        self.render_cache = self.current_question
        return self.render_cache
        
    def step(self, action):
        is_correct, is_valid = self._check_answer(action)
        reward = 1.0 / (2 ** self.step_num) if is_correct else 0.0
        if is_correct:
            observation = "Correct!"
            done = True
        else:
            observation = "Incorrect. Please think again."
            done = False
        self.step_num += 1
        info = {"action_is_valid": is_valid, "success": is_correct}
        self.render_cache = observation
        return self.render_cache, reward, done, info
    
    def _check_answer(self, user_answer):
        """Check if the user's answer matches the correct answer."""
        user_answer = user_answer.strip()
        normalized_answer = re.sub(r'\s+', '', user_answer.lower())
        if self.correct_answer:
            normalized_label = re.sub(r'\s+', '', self.correct_answer.lower())
            is_correct = normalized_answer == normalized_label
        else:
            is_correct = False
        is_valid = normalized_answer != ""
        return is_correct, is_valid

    def render(self):
        return self.render_cache


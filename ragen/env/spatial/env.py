import gymnasium as gym
import numpy as np
from typing import Any, Dict, Tuple, Optional, List
import os

from ragen.env.base import BaseLanguageBasedEnv
from ragen.env.spatial.env_config import SpatialGymConfig
from ragen.env.spatial.Base.tos_base.utils.room_utils import RoomGenerator
from ragen.env.spatial.Base.tos_base.actions.actions import ActionSequence, TermAction, ObserveAction, MoveAction, ActionResult
from ragen.env.spatial.Base.tos_base.evaluation.task_types import EvalTaskType
from ragen.env.spatial.Base.tos_base.managers.exploration_manager import ExplorationManager
from ragen.env.spatial.prompter import SpatialPrompter

class SpatialGym(BaseLanguageBasedEnv):
    def __init__(self, config: SpatialGymConfig = None):
        super().__init__()
        self.config = config or SpatialGymConfig()
        self.room = None
        self.agent = None
        self.current_answer = None
        self.render_mode = self.config.render_mode
        self.max_steps = self.config.max_exp_steps
        self.current_step_count = 0
        self.last_obs = ""
        self.last_info = {}
        
        self.exploration_manager = None
        self.prompter = SpatialPrompter(self.config, np.random.RandomState(42)) # Seed updated in reset

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> str:
        super().reset(seed=seed) # Sets self.np_random
        
        self.prompter.np_random = self.np_random
        
        # Convert eval_tasks (List[str]) to List[Dict] for RoomGenerator validation
        eval_tasks_dicts = [{"task_type": t} for t in self.config.eval_tasks]
        
        # Generate room
        self.room, self.agent = RoomGenerator.generate_room(
            room_size=self.config.room_size,
            n_objects=self.config.n_objects,
            np_random=self.np_random,
            level=self.config.level,
            main=self.config.main,
            eval_tasks=eval_tasks_dicts
        )
        
        # Initialize ExplorationManager
        self.exploration_manager = ExplorationManager(self.room, self.agent)
        
        # Select evaluation task
        task_name = self.np_random.choice(self.config.eval_tasks)
        
        # Create task
        current_task = EvalTaskType.create_task(
            task_name, 
            np_random=self.np_random, 
            room=self.room, 
            agent=self.agent
        )
        
        # Generate question
        current_question = current_task.generate_question()
        self.current_answer = current_task.answer
        
        # Generate initial prompt
        obs_dict = self.prompter.get_initial_observation_prompt(
            self.room, 
            self.agent, 
            question=current_question
        )
        prompt = obs_dict['obs_str']
        
        self.current_step_count = 0
        self.last_obs = prompt
        self.last_info = {}
        return prompt
        
    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        self.current_step_count += 1
        
        # Parse action
        seq = ActionSequence.parse(action)
        if seq is None:
            reward = -1.0
            done = False
            info = {"error": "Invalid action format", "success": False}
            obs = "Invalid action format. Please follow the grammar."
            self.last_obs = obs
            return obs, reward, done, info
            
        # Execute actions using ExplorationManager
        # It handles observed_items, failure fallback to Observe, etc.
        results = self.exploration_manager.execute_action_sequence(seq)
        
        feedback_list = [res.feedback for res in results]
        
        terminated = False
        term_answer = None
        
        # Check for TermAction in results (should be the last one if present and successful)
        # Note: ExplorationManager might execute Observe if motion fails, so check all or last.
        # If TermAction was executed, it means motions succeeded (or there were none).
        for res in results:
            if res.success and res.action_type == 'term':
                terminated = True
                term_answer = res.data.get('answer')
                break

        # Calculate reward
        reward = -0.1
        done = False
        info = {}
        
        if terminated:
            done = True
            if term_answer == self.current_answer:
                reward = 10.0
                info["success"] = True
            else:
                reward = 0.0
                info["success"] = False
            info["answer"] = term_answer
            info["correct_answer"] = self.current_answer
        
        if self.current_step_count >= self.max_steps:
            done = True
            
        obs = "\n".join(feedback_list)
        
        if not done:
            obs += "\n" + self.prompter.get_format_footer(is_exploration=True)
            
        self.last_obs = obs
        self.last_info = info
        
        return obs, reward, done, info

    def render(self):
        return self.last_obs

if __name__ == "__main__":
    env = SpatialGym()
    obs = env.reset(seed=42)
    print("Initial Observation:")
    print(obs)
    from ragen.env.spatial.Base.tos_base.utils.room_utils import RoomPlotter
    RoomPlotter.plot_room(env.room, mode='vision', save_path='room.png')
    
    # Test a few steps
    print("\nStep 1: Move Forward")
    obs, reward, done, info = env.step("move(forward)")
    print(f"Obs: {obs}")
    
    print("\nStep 2: Terminate with wrong answer")
    obs, reward, done, info = env.step("term(yes)")
    print(f"Reward: {reward}, Done: {done}, Info: {info}")

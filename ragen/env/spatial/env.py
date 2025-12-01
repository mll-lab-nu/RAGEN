import gymnasium as gym
import numpy as np
from typing import Any, Dict, Tuple, Optional, List

from ragen.env.base import BaseLanguageBasedEnv
from ragen.env.spatial.config import SpatialGymConfig
from ragen.env.spatial.Base.tos_base.utils.room_utils import RoomGenerator
from ragen.env.spatial.Base.tos_base.actions.actions import ActionSequence, ACTION_REMINDER
from ragen.env.spatial.Base.tos_base.evaluation.task_types import EvalTaskType
from ragen.env.spatial.Base.tos_base.managers.exploration_manager import ExplorationManager
from ragen.env.spatial.prompter import SpatialPrompter

class SpatialGym(BaseLanguageBasedEnv, gym.Env):
    def __init__(self, config: SpatialGymConfig = None):
        super().__init__()
        self.config = config or SpatialGymConfig()
        print(f"Config: {self.config}")
        self.render_mode = self.config.render_mode
        # User requirement: max steps should be 1 step more than max exp steps
        self.max_steps = self.config.max_exp_steps + 1
        
        self.room = None
        self.agent = None
        self.current_answer = None
        self.current_step_count = 0
        self.last_obs = ""
        self.last_info = {}
        
        self.exploration_manager = None
        self.prompter = SpatialPrompter(self.config, np.random.RandomState(42))
        self._rendered = False

    def reset(self, seed: Optional[int] = None, mode=None) -> str:
        gym.Env.reset(self, seed=seed) # Sets self.np_random
        
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
            eval_tasks=eval_tasks_dicts,
            same_room_size=True
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
        prompt = obs_dict['obs_str'] + "\n" + ACTION_REMINDER
        
        self.current_step_count = 0
        self.last_obs = prompt
        self.last_info = {}
        self._rendered = False
        return prompt
        
    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        self._rendered = False
        self.current_step_count += 1
        
        # Parse action
        seq = ActionSequence.parse(action)
        if seq is None:
            return self._step_result(
                obs="Invalid action format." + "\n" + ACTION_REMINDER,
                reward=-1.0,
                done=False,
                info={"error": "Invalid action format", "success": False}
            )
            
        # Execute actions
        results = self.exploration_manager.execute_action_sequence(seq)
        feedback_list = [res.message for res in results]
        
        terminated = False
        term_answer = None
        
        # Check for TermAction
        for res in results:
            if res.success and res.action_type == 'term':
                terminated = True
                term_answer = res.data.get('answer')
                break

        # Calculate reward and done
        reward = -0.1
        done = False
        info = {}
        
        if terminated:
            done = True
            if term_answer == self.current_answer:
                reward = 10.0
                info["success"] = True
            else:
                reward = -1
                info["success"] = False
            info["answer"] = term_answer
            info["correct_answer"] = self.current_answer
        
        if self.current_step_count >= self.max_steps:
            done = True
            
        obs = "\n".join(feedback_list) + "\n" + ACTION_REMINDER

            
        return self._step_result(obs, reward, done, info)

    def _step_result(self, obs, reward, done, info):
        self.last_obs = obs
        self.last_info = info
        return obs, reward, done, info

    def render(self, mode=None):
        if self._rendered:
            return "invalid format" + "\n" + ACTION_REMINDER
        self._rendered = True
        return self.last_obs
        
    def close(self):
        pass

if __name__ == "__main__":
    config = SpatialGymConfig(room_size=[20, 20], n_objects=5, level=0, main=6)
    env = SpatialGym(config)
    obs = env.reset(seed=42)
    print("Initial Observation:")
    print(obs)

    from ragen.env.spatial.Base.tos_base.utils.room_utils import RoomPlotter
    RoomPlotter.plot(env.room, env.agent, mode='img', save_path='room.png')
    
    # Test a few steps
    print("\nStep 0: Invalid action")
    obs, reward, done, info = env.step("Actions: [Rotate(45)]")
    obs = env.render()
    print(f"Reward: {reward}, Done: {done}, Info: {info}, Obs: {obs}")

    print("\nStep 1: Rotate and Observe")
    obs, reward, done, info = env.step("Actions: [Rotate(90), Observe()]")
    obs = env.render()
    print(f"Reward: {reward}, Done: {done}, Info: {info}, Obs: {obs}")
    
    print("\nStep 2: Jump to red door and Observe")
    obs, reward, done, info = env.step("Actions: [JumpTo(red door), Observe()]")
    obs = env.render()
    print(f"Reward: {reward}, Done: {done}, Info: {info}, Obs: {obs}")
    
    print("\nStep 3: Terminate with answer")
    obs, reward, done, info = env.step("Actions: [Term(C)]")
    obs = env.render()
    print(f"Reward: {reward}, Done: {done}, Info: {info}, Obs: {obs}")
    
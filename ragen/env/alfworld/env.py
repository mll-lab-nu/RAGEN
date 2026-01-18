"""
This is the environment for the ALFRED dataset.
author: Qineng Wang
date: 2025-03-30

Modified to match verl-agent AlfWorld implementation:
- Dynamic mode switching between train/val
- Simplified reward: score * won
- Admissible actions included in observation
"""
import os
import random
import textworld
import textworld.gym
import numpy as np
from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv, AlfredDemangler, AlfredInfos
from ragen.env.base import BaseLanguageBasedEnv
from ragen.env.alfworld.config import AlfredEnvConfig
from ragen.env.alfworld.utils import load_config

_RAW_ENV_CACHE = {}

class AlfredTXTEnv(BaseLanguageBasedEnv):

    # Mode mapping: RAGEN mode -> AlfWorld train_eval
    MODE_MAP = {
        'train': 'train',
        'val': None,  # Will use config.eval_dataset
        'test': None,  # Will use config.eval_dataset
    }

    def __init__(self, config: AlfredEnvConfig = AlfredEnvConfig(), mode='train'):
        """
        Initialize AlfWorld environment.

        Args:
            config: AlfredEnvConfig instance
            mode: 'train' or 'val' - determines which dataset split to use
        """
        super().__init__()
        self.config = config
        self.raw_env_config = load_config(self.config.config_file)

        # Map RAGEN mode to AlfWorld train_eval
        self.current_mode = mode
        self._alfworld_mode = self._get_alfworld_mode(mode)

        # Initialize raw environment
        self.raw_env = self._get_cached_raw_env(self._alfworld_mode)
        self.num_games = self.raw_env.num_games
        self.game_files = list(self.raw_env.game_files)

        self.alfred_env = None
        self.current_game_file = None
        self.render_cache = None
        self.available_actions = None
        self.render_mode = self.config.render_mode
        assert self.render_mode == 'text'

    def _get_alfworld_mode(self, mode: str) -> str:
        """Convert RAGEN mode to AlfWorld train_eval string."""
        if mode == 'train':
            return 'train'
        else:
            # Use eval_dataset from config for val/test modes
            return self.config.eval_dataset

    def _get_cached_raw_env(self, alfworld_mode: str) -> AlfredTWEnv:
        cache_key = (os.path.abspath(self.config.config_file), alfworld_mode)
        raw_env = _RAW_ENV_CACHE.get(cache_key)
        if raw_env is None:
            raw_env = AlfredTWEnv(config=self.raw_env_config, train_eval=alfworld_mode)
            _RAW_ENV_CACHE[cache_key] = raw_env
        return raw_env
    
    def _reinitialize_for_mode(self, mode: str):
        """Reinitialize the environment when mode changes."""
        new_alfworld_mode = self._get_alfworld_mode(mode)

        if new_alfworld_mode != self._alfworld_mode:
            # Close existing environment
            if hasattr(self, 'alfred_env') and self.alfred_env is not None:
                try:
                    self.alfred_env.close()
                except Exception:
                    pass

            # Reinitialize with new mode
            self._alfworld_mode = new_alfworld_mode
            self.current_mode = mode
            self.raw_env = self._get_cached_raw_env(self._alfworld_mode)
            self.num_games = self.raw_env.num_games
            self.game_files = list(self.raw_env.game_files)
            self.alfred_env = None

    def reset(self, seed=None, mode=None):
        """
        Reset the environment with a specific seed.

        Args:
            seed: Random seed for game selection
            mode: 'train', 'val', or 'test' - if different from current mode,
                  reinitializes the environment with the new dataset split

        Returns:
            observation: Initial observation text with admissible actions
        """
        try:
            # Handle mode switching dynamically
            if mode is not None and mode != self.current_mode:
                self._reinitialize_for_mode(mode)
            elif mode is not None:
                self.current_mode = mode

            # Select game based on mode
            if self.current_mode == "test":
                if seed is None:
                    raise ValueError("Seed must be provided in test mode.")
                selected_game = self.game_files[seed % len(self.game_files)]
            else:
                if seed is not None:
                    np.random.seed(seed)
                    random.seed(seed)
                    game_idx = seed % len(self.game_files)
                    selected_game = self.game_files[game_idx]
                else:
                    selected_game = random.choice(self.game_files)

            self.current_game_file = selected_game

            if hasattr(self, 'alfred_env') and self.alfred_env is not None:
                try:
                    self.alfred_env.close()
                except Exception:
                    pass

            request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
            wrappers = [AlfredDemangler(), AlfredInfos()]
            max_steps = self.raw_env_config["rl"]["training"]["max_nb_steps_per_episode"]

            env_id = textworld.gym.register_game(
                selected_game,
                request_infos=request_infos,
                batch_size=1,
                asynchronous=False,
                max_episode_steps=max_steps,
                wrappers=wrappers
            )

            self.alfred_env = textworld.gym.make(env_id)

            obs, info = self.alfred_env.reset()
            self.available_actions = info["admissible_commands"][0]
            self.instruction_text = obs[0]
            # Include admissible actions in render cache
            self.render_cache = self._format_observation(obs[0], self.available_actions)
            return self.render_cache

        except (RuntimeError, RuntimeWarning) as e:
            print(f"Error in reset: {e}")
            next_seed = abs(hash(str(seed))) % (2 ** 32) if seed is not None else None
            return self.reset(next_seed, mode=self.current_mode)

    def _format_observation(self, obs: str, admissible_actions: list) -> str:
        """Format observation to include admissible actions."""
        actions_str = ", ".join(admissible_actions)
        return f"{obs}\n\nAdmissible actions: [{actions_str}]"
    
    def step(self, action: str):
        """
        Take a step in the environment using the provided action string.

        Reward scheme (matching verl-agent):
        - reward = config.score * won (10.0 on success, 0.0 otherwise)

        Args:
            action: The action string to execute

        Returns:
            observation: Updated observation with admissible actions
            reward: Sparse reward (score on success, 0 otherwise)
            done: Whether episode is finished
            info: Additional information dict
        """
        action_is_available = action in self.available_actions

        # Execute action in environment
        obs, _, dones, infos = self.alfred_env.step([action])
        observation = obs[0]
        self.available_actions = infos["admissible_commands"][0]
        done = dones[0]
        won = infos["won"][0]

        # Simplified reward matching verl-agent: score * won
        reward = self.config.score * float(won)

        # Format observation with admissible actions
        self.render_cache = self._format_observation(observation, self.available_actions)

        info = {
            "action_is_effective": True,
            "action_is_valid": action_is_available,
            "success": won
        }

        return self.render_cache, reward, done, info
    
    def render(self):
        """Return current observation with admissible actions."""
        return self.render_cache

    def close(self):
        """Clean up environment resources."""
        self.render_cache = None
        if hasattr(self, 'alfred_env') and self.alfred_env is not None:
            try:
                self.alfred_env.close()
            except Exception:
                pass


if __name__ == "__main__":
    import os
    os.environ["ALFWORLD_DATA"] = os.path.expanduser("~/.cache/alfworld")

    # Test basic environment functionality
    print("\n=== Test 1: Basic environment with train mode ===")
    env = AlfredTXTEnv()
    obs = env.reset(seed=42, mode='train')
    print(f"Train observation (first 500 chars):\n{obs[:500]}...")

    # Test mode switching
    print("\n=== Test 2: Mode switching to val ===")
    obs = env.reset(seed=42, mode='val')
    print(f"Val observation (first 500 chars):\n{obs[:500]}...")

    # Test step with admissible action
    print("\n=== Test 3: Taking a step ===")
    if env.available_actions:
        action = env.available_actions[0]
        print(f"Taking action: {action}")
        obs, reward, done, info = env.step(action)
        print(f"Reward: {reward}, Done: {done}, Info: {info}")
        print(f"New observation (first 300 chars):\n{obs[:300]}...")

    env.close()
    print("\n=== All tests completed ===")
    

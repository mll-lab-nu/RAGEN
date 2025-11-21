import numpy as np
from typing import Optional, Any
from ragen.env.spatial.Base.tos_base.prompts.prompter import Prompter
from ragen.env.spatial.Base.tos_base.actions.actions import ActionSequence
from ragen.env.spatial.Base.tos_base.utils.room_utils import get_room_description
from ragen.env.spatial.Base.tos_base.core.relationship import (
    PairwiseRelationship, PairwiseRelationshipDiscrete, ProximityRelationship, DegreeRel, OrientationRel
)
from ragen.env.spatial.Base.tos_base.prompts.prompts import (
    INSTRUCTION_TEMPLATE_TEXT, SHARED_INTRO_TEXT,
    SHARED_MULTIROOM_RULES, SHARED_RULES_COMMON, ACTIVE_RULES_EXTRA
)

class SpatialPrompter(Prompter):
    def __init__(self, config, np_random: np.random.RandomState):
        # Initialize without image_handler
        super().__init__(config, np_random, image_handler=None)

    def get_initial_observation_prompt(
            self,
            room,
            agent,
            question: str,
            exp_history = None
        ) -> dict:
        """
        Generates the initial observation prompt.
        Forces active exploration instructions and sets the goal to answering the evaluation question.
        Removes all vision-related logic.
        """
        obs = {}
        topdown = self.config.prompt_config['topdown']

        room_desc = get_room_description(room, agent, with_topdown=topdown)

        observation_instructions = (
            PairwiseRelationship.prompt()
            + f"\n{DegreeRel.prompt()}"
            + f"\n{OrientationRel.prompt()}"
            + f"\n{PairwiseRelationshipDiscrete.prompt()}"
            + f"\n{ProximityRelationship.prompt()}"
        )

        # Always include action instructions (text only)
        exp_instructions = f"Action Instructions:\n{ActionSequence.get_usage_instructions(vision=False)}"
        exp_instructions += f"\n\nYou have a maximum of {self.config.max_exp_steps} exploration steps."

        template = INSTRUCTION_TEMPLATE_TEXT

        # Custom goal
        goal_lines = "Goal: Answer the evaluation question."

        fmt_kwargs = {
            'title': 'Spatial Exploration Task',
            'intro': SHARED_INTRO_TEXT,
            'goal_lines': goal_lines,
            'format_rules': self._build_format_rules(is_exploration=True), # Use exploration format (Actions: [...])
            'observation_instructions': observation_instructions,
            'exp_instructions': exp_instructions,
            'room_info': room_desc,
            'multiroom_rules': SHARED_MULTIROOM_RULES,
            'active_rules_extra': ACTIVE_RULES_EXTRA,
            'rules_common': SHARED_RULES_COMMON,
            'exp_history': "", # Initial prompt has no history
        }

        obs_str = template.format(**fmt_kwargs)
        
        # Append evaluation question
        if question:
             obs_str += f"\n## Evaluation Question\n{question}"

        obs['obs_str'] = obs_str + "\n" + self.get_format_footer(is_exploration=True)
        return obs

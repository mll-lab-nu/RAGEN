# ragen/env/spatial/prompts.py

SHARED_INTRO_TEXT = "You are a spatial reasoner in indoor environment, a 2D, text-only N×M grid. Every object including you is a point at integer (x, y) coordinates."

SHARED_INTRO_VISION = (
    "You are a spatial reasoner in a 3D simulated environment. "
    "The world is rendered in 3D but abstracted into a discrete 2D grid of size N×M. "
    "Every entity, including yourself, is represented by integer coordinates (x, y) on this grid."
)

SHARED_MULTIROOM_RULES = """\
Multi-room rules (may exist multiple rooms):
- Your vision is confined to your current room.
- Doors block vision between rooms.
- Exception: When located in a doorway, door is open and invisible, you can see into both connected rooms.
- Rooms connect via doors on vertical (front/back) or horizontal (left/right) walls.
"""

SHARED_RULES_COMMON = """\
- Field of view: 90°
"""

# Optimized: Emphasize answering evaluation question, remove coverage goals.
ACTIVE_RULES_EXTRA = ""

VISION_EXAMPLE = """\
Here is an example of your observation: blue cylinder 1 m straight ahead; red cylinder 2 m straight ahead; yellow cylinder 2 m at 45° to your front-left; green cylinder 3 m at 22.5° to your front-slight-right:
{image_placeholder}

The image shows all objects in the room. Each tile is numbered (1-N) in the top-left, matching the object order in the room layout.
For items with a facing direction, two copies are shown side-by-side: the left copy has its front facing the camera; the right copy has its front facing left.
Items without a meaningful facing direction are shown once.
{image_placeholder}
"""

_BASE_TEMPLATE = """\
# {title}

{intro}

{goal_lines}

{multiroom_rules}

Relationship instructions:
{observation_instructions}

{exp_instructions}

{format_rules}

Rules:
{active_rules_extra}{rules_common}

Room Layout and initial state:
{room_info}
"""

INSTRUCTION_TEMPLATE_TEXT = _BASE_TEMPLATE + """
{exp_history}
"""

INSTRUCTION_TEMPLATE_VISION = _BASE_TEMPLATE + """
{vision_example}

{exp_history}
"""

EVALUATION_INSTRUCTION = "{eval_question}"
SHORT_EXPLORATION_PROMPT = "Please respond with valid actions to explore the rooms."
SHORT_EVALUATION_PROMPT = "Please respond with a valid answer to the question."

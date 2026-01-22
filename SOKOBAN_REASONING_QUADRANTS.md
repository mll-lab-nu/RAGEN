# Sokoban Reasoning Quadrants

This document outlines the four types of reasoning behaviors observed in Large Language Models (LLMs) when interacting with environments like Sokoban. These behaviors are categorized by two axes: **Mutual Information** (Relevance to the specific task) and **Entropy** (Diversity of the reasoning chain).

We illustrate each quadrant with specific Sokoban scenarios generated in `scenarios_tailored/`.

## The Axes
*   **X-Axis: Mutual Information ($I(X; Z)$)**: How well the reasoning correlates with the specific input/prompt. High = Relevant. Low = Irrelevant.
*   **Y-Axis: Entropy ($H(Z|X)$)**: The diversity and richness of the generated reasoning. High = Diverse. Low = Repetitive/Robotic.

---

## 1. True Diverse Reasoning (The Goal)
**Characteristics:** High Mutual Information, High Entropy.
The model generates rich, step-by-step reasoning that is directly applicable to the current board state. It "thinks" like a human solving the puzzle.

### Scenario 1: "Move up agent twice..."
*   **Description:** A scenario requiring multi-step planning where the specific moves are articulated.
*   **Text Example:** "Move up agent twice..." / "I see two boxes and..."
*   **Image:** ![TrueDiverseReasoning1](/home/deimos/RAGEN/scenarios_tailored/TrueDiverseReasoning1.png)

### Scenario 2: "I see box is on top of..."
*   **Description:** A layout where spatial relationships (above/below) are key.
*   **Text Example:** "I see box is on top of..." / "Move the box down then..."
*   **Image:** ![TrueDiverseReasoning2](/home/deimos/RAGEN/scenarios_tailored/TrueDiverseReasoning2.png)

---

## 2. Template Collapse
**Characteristics:** Low Mutual Information, High Entropy.
The model produces long, detailed, and varied text, but it is generic advice that could apply to *any* Sokoban level. It ignores the specific constraints of the current map.

### Scenario 1: Easy to get stuck
*   **Description:** A cramped level with corners where generic advice like "don't block corners" is relevant but unhelpful if not specific.
*   **Text Example:** "Avoid getting stuck..." / "Plan your steps carefully..."
*   **Image:** ![TemplateCollapse1](/home/deimos/RAGEN/scenarios_tailored/TemplateCollapse1.png)

### Scenario 2: Hard to get stuck
*   **Description:** A wide-open level where "avoid corners" is irrelevant advice, highlighting the disconnect.
*   **Text Example:** "That's a good question..."
*   **Image:** ![TemplateCollapse2](/home/deimos/RAGEN/scenarios_tailored/TemplateCollapse2.png)

---

## 3. Echo Trap
**Characteristics:** Low Mutual Information, Low Entropy.
The model fails to generate meaningful content and falls into repetitive loops or simple acknowledgments.

### Scenario 1 & 2: Diverse Layouts
*   **Description:** Standard random layouts where the model simply fails to engage.
*   **Text Example:** "I need to solve this task." (Repeatedly)
*   **Image 1:** ![EchoTrap1](/home/deimos/RAGEN/scenarios_tailored/EchoTrap1.png)
*   **Image 2:** ![EchoTrap2](/home/deimos/RAGEN/scenarios_tailored/EchoTrap2.png)

---

## 4. Strategic Compression
**Characteristics:** High Mutual Information, Low Entropy.
The model solves the task efficiently but outputs only the raw solution or highly compressed instructions, lacking the "reasoning process" or explanation.

### Scenario 1: "Go up twice and left once"
*   **Description:** A specific puzzle where the exact solution is a short sequence.
*   **Text Example:** "Go up twice and left once."
*   **Image:** ![StrategicCompression1](/home/deimos/RAGEN/scenarios_tailored/StrategicCompression1.png)

### Scenario 2: "Go down twice then done"
*   **Description:** A specific puzzle solvable by pushing a box down.
*   **Text Example:** "Go down twice then done."
*   **Image:** ![StrategicCompression2](/home/deimos/RAGEN/scenarios_tailored/StrategicCompression2.png)

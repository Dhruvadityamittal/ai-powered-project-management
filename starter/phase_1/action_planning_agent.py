import os
from dotenv import load_dotenv

from workflow_agents.base_agents import (
    ActionPlanningAgent,
)


load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not set."
    )

knowledge = """
A technical project plan can be broken into
three major activities:

1. Define user stories.
2. Group stories into product features.
3. Define engineering tasks required to implement
   the stories and features.
"""

agent = ActionPlanningAgent(
    openai_api_key,
    knowledge,
)

prompt = (
    "Create a development plan for a new email router."
)

steps = agent.extract_steps_from_prompt(prompt)

print("\nWorkflow steps:")

for index, step in enumerate(
    steps,
    start=1,
):
    print(f"{index}. {step}")
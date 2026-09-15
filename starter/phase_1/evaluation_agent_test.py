import os
from dotenv import load_dotenv

from workflow_agents.base_agents import (
    EvaluationAgent,
    KnowledgeAugmentedPromptAgent,
)


load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not set."
    )

worker_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key,
    "a product manager",
    """
    A user story must follow:

    As a [user],
    I want [action],
    so that [benefit].
    """,
)

evaluation_agent = EvaluationAgent(
    openai_api_key,
    "You are an evaluator.",
    """
    The answer must contain user stories that
    follow the required As a / I want / so that format.
    """,
    worker_agent,
    3,
)

prompt = (
    "Write two user stories for an email router."
)

print("Testing EvaluationAgent")

result = evaluation_agent.evaluate(prompt)

print("\nFinal result:")
print(result["final_response"])

print(
    "\nEvaluationAgent test completed successfully."
)
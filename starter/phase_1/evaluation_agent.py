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

worker = KnowledgeAugmentedPromptAgent(
    openai_api_key,
    "a product manager",
    """
    User stories must use this structure:

    As a [type of user],
    I want [an action],
    so that [a benefit].
    """,
)

evaluator = EvaluationAgent(
    openai_api_key,
    "You are a strict product-management evaluator.",
    """
    The response must contain user stories following:

    As a [type of user],
    I want [an action],
    so that [a benefit].
    """,
    worker,
    3,
)

prompt = (
    "Create three user stories for an email routing system."
)

result = evaluator.evaluate(prompt)

print("\nFinal response:")
print(result["final_response"])

print(
    f"\nAccepted: {result['accepted']}"
)

print(
    f"Iterations: {result['iterations']}"
)
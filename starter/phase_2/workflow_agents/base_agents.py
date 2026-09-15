"""Reusable agent classes for the AI-powered project-management workflow."""

import csv
import os
import re
import uuid
from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd
from openai import OpenAI


def _client(api_key: str) -> OpenAI:
    """Create an OpenAI client.

    OPENAI_BASE_URL is optional. The Udacity/Vocareum environment normally
    supplies a compatible endpoint; standard OpenAI users can omit it.
    """
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


class DirectPromptAgent:
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key

    def respond(self, prompt: str) -> str:
        client = _client(self.openai_api_key)

        response = client.chat.completions.create(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-3.5-turbo"),
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0,
        )

        return response.choices[0].message.content.strip()


class AugmentedPromptAgent:
    def __init__(self, openai_api_key: str, persona: str):
        self.openai_api_key = openai_api_key
        self.persona = persona

    def respond(self, input_text: str) -> str:
        client = _client(self.openai_api_key)

        response = client.chat.completions.create(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-3.5-turbo"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are {self.persona}. "
                        "Forget all previous context. "
                        "Follow the persona when answering the user's prompt."
                    ),
                },
                {
                    "role": "user",
                    "content": input_text,
                },
            ],
            temperature=0,
        )

        return response.choices[0].message.content.strip()


class KnowledgeAugmentedPromptAgent:
    def __init__(
        self,
        openai_api_key: str,
        persona: str,
        knowledge: str,
    ):
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.knowledge = knowledge

    def respond(self, input_text: str) -> str:
        client = _client(self.openai_api_key)

        system_prompt = (
            f"You are {self.persona} knowledge-based assistant. "
            "Forget all previous context.\n\n"
            "Use only the following knowledge to answer. "
            "Do not use your own knowledge:\n\n"
            f"{self.knowledge}\n\n"
            "Answer the prompt based on this knowledge, not your own."
        )

        response = client.chat.completions.create(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-3.5-turbo"),
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": input_text,
                },
            ],
            temperature=0,
        )

        return response.choices[0].message.content.strip()


class RAGKnowledgePromptAgent:
    """Retrieval-augmented generation agent using embeddings over local chunks."""

    def __init__(
        self,
        openai_api_key,
        persona,
        chunk_size=2000,
        chunk_overlap=100,
    ):
        self.persona = persona
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_api_key = openai_api_key

        self.unique_filename = (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
            f"{uuid.uuid4().hex[:8]}.csv"
        )

    def get_embedding(self, text):
        client = _client(self.openai_api_key)

        response = client.embeddings.create(
            model=os.getenv(
                "OPENAI_EMBEDDING_MODEL",
                "text-embedding-3-large",
            ),
            input=text,
            encoding_format="float",
        )

        return response.data[0].embedding

    def calculate_similarity(self, vector_one, vector_two):
        vec1 = np.asarray(vector_one)
        vec2 = np.asarray(vector_two)

        denominator = np.linalg.norm(vec1) * np.linalg.norm(vec2)

        if denominator == 0:
            return 0.0

        return float(np.dot(vec1, vec2) / denominator)

    def chunk_text(self, text):
        text = re.sub(r"[ \t]+", " ", text).strip()

        if len(text) <= self.chunk_size:
            chunks = [
                {
                    "chunk_id": 0,
                    "text": text,
                    "chunk_size": len(text),
                }
            ]
        else:
            chunks = []
            start = 0
            chunk_id = 0

            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                candidate = text[start:end]

                if "\n" in candidate:
                    end = start + candidate.rfind("\n") + 1

                if end <= start:
                    end = min(start + self.chunk_size, len(text))

                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "text": text[start:end],
                        "chunk_size": end - start,
                        "start_char": start,
                        "end_char": end,
                    }
                )

                if end == len(text):
                    break

                start = max(0, end - self.chunk_overlap)
                chunk_id += 1

        filename = f"chunks-{self.unique_filename}"

        with open(
            filename,
            "w",
            newline="",
            encoding="utf-8",
        ) as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=["text", "chunk_size"],
            )

            writer.writeheader()

            for chunk in chunks:
                writer.writerow(
                    {
                        "text": chunk["text"],
                        "chunk_size": chunk["chunk_size"],
                    }
                )

        return chunks

    def calculate_embeddings(self):
        filename = f"chunks-{self.unique_filename}"

        if not os.path.exists(filename):
            raise FileNotFoundError(
                f"{filename} not found. "
                "Call chunk_text() before calculate_embeddings()."
            )

        df = pd.read_csv(
            filename,
            encoding="utf-8",
        )

        df["embeddings"] = df["text"].apply(
            self.get_embedding
        )

        output_filename = (
            f"embeddings-{self.unique_filename}"
        )

        df.to_csv(
            output_filename,
            encoding="utf-8",
            index=False,
        )

        return df

    def find_prompt_in_knowledge(self, prompt):
        filename = f"embeddings-{self.unique_filename}"

        if not os.path.exists(filename):
            raise FileNotFoundError(
                f"{filename} not found. "
                "Call calculate_embeddings() before retrieval."
            )

        prompt_embedding = self.get_embedding(prompt)

        df = pd.read_csv(
            filename,
            encoding="utf-8",
        )

        df["embeddings"] = df["embeddings"].apply(
            lambda x: np.asarray(eval(x))
        )

        df["similarity"] = df["embeddings"].apply(
            lambda emb: self.calculate_similarity(
                prompt_embedding,
                emb,
            )
        )

        best_chunk = df.loc[
            df["similarity"].idxmax(),
            "text",
        ]

        client = _client(self.openai_api_key)

        response = client.chat.completions.create(
            model=os.getenv(
                "OPENAI_CHAT_MODEL",
                "gpt-3.5-turbo",
            ),
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are {self.persona}, "
                        "a knowledge-based assistant. "
                        "Forget previous context."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Answer based only on this retrieved information:\n"
                        f"{best_chunk}\n\n"
                        f"Prompt: {prompt}"
                    ),
                },
            ],
            temperature=0,
        )

        return response.choices[0].message.content.strip()


class EvaluationAgent:
    def __init__(
        self,
        openai_api_key: str,
        persona: str,
        evaluation_criteria: str,
        worker_agent: Any,
        max_interactions: int = 3,
    ):
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.agent_to_evaluate = worker_agent
        self.max_interactions = max(
            1,
            max_interactions,
        )

    def evaluate(self, initial_prompt: str) -> Dict[str, Any]:
        client = _client(self.openai_api_key)

        prompt_to_evaluate = initial_prompt
        response_from_worker = ""
        evaluation = ""
        iterations = 0

        for i in range(self.max_interactions):
            iterations = i + 1

            print(
                f"\n--- Interaction {iterations} ---"
            )

            print(
                "Step 1: Worker agent generates "
                "a response to the prompt"
            )

            print(
                f"Prompt:\n{prompt_to_evaluate}"
            )

            response_from_worker = (
                self.agent_to_evaluate.respond(
                    prompt_to_evaluate
                )
            )

            print(
                "Worker Agent Response:"
            )

            print(response_from_worker)

            print(
                "Step 2: Evaluator agent "
                "judges the response"
            )

            eval_prompt = (
                "Does the following answer:\n"
                f"{response_from_worker}\n\n"
                "Meet this criteria:\n"
                f"{self.evaluation_criteria}\n\n"
                "Respond Yes or No first, "
                "followed by a concise reason."
            )

            response = client.chat.completions.create(
                model=os.getenv(
                    "OPENAI_CHAT_MODEL",
                    "gpt-3.5-turbo",
                ),
                messages=[
                    {
                        "role": "system",
                        "content": self.persona,
                    },
                    {
                        "role": "user",
                        "content": eval_prompt,
                    },
                ],
                temperature=0,
            )

            evaluation = (
                response.choices[0]
                .message.content
                .strip()
            )

            print(
                "Evaluator Agent Evaluation:"
            )

            print(evaluation)

            if evaluation.lower().startswith("yes"):
                print(
                    "✅ Final solution accepted."
                )
                break

            if iterations == self.max_interactions:
                print(
                    "⚠️ Maximum interactions reached."
                )
                break

            print(
                "Step 3: Generate correction instructions"
            )

            instruction_prompt = (
                "Provide concise correction instructions "
                "for the worker agent based on the following "
                "evaluation. Do not generate the final answer. "
                "Only explain what should be corrected.\n\n"
                f"Evaluation:\n{evaluation}"
            )

            instruction_response = (
                client.chat.completions.create(
                    model=os.getenv(
                        "OPENAI_CHAT_MODEL",
                        "gpt-3.5-turbo",
                    ),
                    messages=[
                        {
                            "role": "system",
                            "content": self.persona,
                        },
                        {
                            "role": "user",
                            "content": instruction_prompt,
                        },
                    ],
                    temperature=0,
                )
            )

            correction = (
                instruction_response.choices[0]
                .message.content
                .strip()
            )

            print(
                f"Correction instructions:\n{correction}"
            )

            prompt_to_evaluate = (
                f"{initial_prompt}\n\n"
                "Previous response did not meet the criteria. "
                "Correct it using these instructions:\n"
                f"{correction}"
            )

        return {
            "final_response": response_from_worker,
            "evaluation": evaluation,
            "iterations": iterations,
            "accepted": evaluation.lower().startswith(
                "yes"
            ),
        }


class RoutingAgent:
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.agents = []

    def get_embedding(self, text: str):
        client = _client(self.openai_api_key)
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float",
        )
        return response.data[0].embedding

    def calculate_similarity(self, vector_one, vector_two):
        vec1 = np.asarray(vector_one)
        vec2 = np.asarray(vector_two)
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    def route(self, query: str):
        if not self.agents:
            raise ValueError("No agents have been configured for the RoutingAgent.")

        query_embedding = self.get_embedding(query)
        best_agent = None
        best_score = -1.0

        for agent in self.agents:
            description_embedding = self.get_embedding(agent["description"])
            score = self.calculate_similarity(query_embedding, description_embedding)
            if score > best_score:
                best_score = score
                best_agent = agent

        print(f"Routing task to: {best_agent['name']} with similarity {best_score:.4f}")
        return best_agent["func"](query)


class ActionPlanningAgent:
    def __init__(
        self,
        openai_api_key: str,
        knowledge: str,
    ):
        self.openai_api_key = openai_api_key
        self.knowledge = knowledge

    def extract_steps_from_prompt(self, prompt: str):
        client = _client(self.openai_api_key)

        system_prompt = (
            "You are an action planning agent. "
            "Break a high-level project-management request "
            "into a logical sequence of actionable sub-tasks.\n\n"
            "Use the following knowledge:\n"
            f"{self.knowledge}\n\n"
            "Return ONLY a numbered list of steps. "
            "Each step should be independently actionable "
            "and suitable for routing to a specialist."
        )

        response = client.chat.completions.create(
            model=os.getenv(
                "OPENAI_CHAT_MODEL",
                "gpt-3.5-turbo",
            ),
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
        )

        raw_steps = (
            response.choices[0]
            .message.content
            .strip()
        )

        steps = []

        for line in raw_steps.splitlines():
            cleaned = re.sub(
                r"^\s*(?:[-*]|\d+[.)])\s*",
                "",
                line,
            ).strip()

            if cleaned:
                steps.append(cleaned)

        return steps
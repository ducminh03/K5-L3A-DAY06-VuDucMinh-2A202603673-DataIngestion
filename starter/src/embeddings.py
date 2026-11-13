"""Choose a real embedding provider (API or local) and implement this contract."""
import time
from typing import Protocol

from google import genai


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class MyEmbedder:
    """Generate Gemini embeddings, batching document inputs for ingestion."""

    def __init__(self, model_name: str, dimensions: int = 3072) -> None:
        if not model_name:
            raise ValueError("model_name must not be empty")
        if dimensions <= 0:
            raise ValueError("dimensions must be a positive integer")

        self.model_name = model_name
        self.dimensions = dimensions
        self._client = genai.Client()

    def _format_input(
        self, text: str, task_type: str
    ) -> tuple[str, dict[str, int | str]]:
        config: dict[str, int | str] = {"output_dimensionality": self.dimensions}
        if self.model_name.endswith("gemini-embedding-2"):
            if task_type == "RETRIEVAL_DOCUMENT":
                text = f"title: none | text: {text}"
            else:
                text = f"task: question answering | query: {text}"
        else:
            config["task_type"] = task_type
        return text, config

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise ValueError("texts must not contain empty strings")

        requests = []
        for text in texts:
            contents, config = self._format_input(text, "RETRIEVAL_DOCUMENT")
            requests.append({
                "contents": contents,
                "config": config,
            })
        job = self._client.batches.create_embeddings(
            model=self.model_name,
            src={"inlined_requests": requests},
            config={"display_name": f"{self.model_name}-document-embeddings"},
        )
        if not job.name:
            raise RuntimeError("Google GenAI created an embedding batch without a job name")

        terminal_states = {
            "JOB_STATE_SUCCEEDED",
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
            "JOB_STATE_EXPIRED",
        }
        while True:
            state = job.state.name if job.state is not None else None
            if state in terminal_states:
                break
            time.sleep(5)
            job = self._client.batches.get(name=job.name)

        if state != "JOB_STATE_SUCCEEDED":
            raise RuntimeError(f"Embedding batch {job.name!r} ended with state {state!r}")

        if job.dest is None or job.dest.inlined_responses is None:
            raise RuntimeError(f"Embedding batch {job.name!r} returned no inline responses")
        responses = job.dest.inlined_responses
        if len(responses) != len(texts):
            raise RuntimeError(
                f"Embedding batch returned {len(responses)} responses for {len(texts)} inputs"
            )

        vectors: list[list[float]] = []
        for response in responses:
            if response.error is not None:
                raise RuntimeError(f"Embedding batch request failed: {response.error}")
            if response.response is None or not response.response.embeddings:
                raise RuntimeError("Embedding batch returned a response without an embedding")
            values = response.response.embeddings[0].values
            if values is None or len(values) != self.dimensions:
                actual_dimensions = len(values) if values is not None else 0
                raise RuntimeError(
                    f"Embedding has {actual_dimensions} dimensions; expected {self.dimensions}"
                )
            vectors.append(values)
        return vectors

    def embed_query(self, question: str) -> list[float]:
        if not question.strip():
            raise ValueError("question must not be empty")

        contents, config = self._format_input(question, "RETRIEVAL_QUERY")
        result = self._client.models.embed_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )
        if not result.embeddings or result.embeddings[0].values is None:
            raise RuntimeError("Google GenAI returned no embedding for the query")

        vector = result.embeddings[0].values
        if len(vector) != self.dimensions:
            raise RuntimeError(
                f"Query embedding has {len(vector)} dimensions; expected {self.dimensions}"
            )
        return vector

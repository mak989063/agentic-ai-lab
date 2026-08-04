from ollama import chat


class OllamaClient:
    """Simple client for interacting with an Ollama chat model."""

    def __init__(self, model: str = "qwen2.5:7b") -> None:
        """Initialize the client with a model name."""
        self.model = model

    def ask(self, prompt: str) -> str:
        """
        Send a prompt to the model and return its response.

        Args:
            prompt: The user's input prompt.

        Returns:
            The model's response as a string.

        Raises:
            RuntimeError: If the model returns an empty response.
        """

        response = chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        content = response.message.content

        if content is None:
            raise RuntimeError("Model returned an empty response.")

        return content


def main() -> None:
    """Application entry point."""

    client = OllamaClient()

    prompt = "Explain Docker in one sentence."

    answer = client.ask(prompt)

    print("\n🤖 Response:\n")
    print(answer)


if __name__ == "__main__":
    main()

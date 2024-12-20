from threading import Lock

import modal
from pathlib import Path

HERE = Path(__file__).parent.resolve()
MODEL_DIR = "/model"
MODEL_NAME = "Llama-3.2-1B-Instruct-Q4_0.gguf"

image = (
    # this gives a glibc issue, so we use ubuntu instead
    # modal.Image.debian_slim(python_version="3.11")
    modal.Image.from_registry("ubuntu:22.04", add_python="3.11")
    .pip_install(
        "hf_transfer",
        "fastapi[standard]",
        "gpt4all==2.8.2",
        "nvidia-cudnn-cu11==8.9.6.50",
        "nvidia-cudnn-cu12==9.1.1.17",
    )
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu116")
    .env(
        {
            "LD_LIBRARY_PATH": "/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib",  # noqa: E501
        }
    )
)

app = modal.App("solara-llm-demo", image=image)


@app.cls(
    gpu="a10g",  # Try using an A100 or H100 if you've got a large model or need big batches!
    concurrency_limit=2,  # default max GPUs for Modal's free tier
)
class Model:
    @modal.build()
    def download_model(self):
        """Download model to cache during modal docker image build."""
        from gpt4all import GPT4All

        # Just to download into the cache
        GPT4All.retrieve_model(MODEL_NAME, verbose=True)
        print("Downloaded model: ", MODEL_NAME)

    @modal.enter()
    def load_model(self):
        """One-time load model during server startup."""
        from gpt4all import GPT4All

        self.model = GPT4All(MODEL_NAME, allow_download=False)
        self.mutex = Lock()

    @modal.method(is_generator=True)
    def predict(self, prompt: str, history: list[dict], temperature: float = 0.1):
        """Predict on the user's output."""
        # handle one prompt at a time, otherwise histories will mix
        with self.model.chat_session(), self.mutex:
            self.model._history = history
            for char in self.model.generate(prompt, streaming=True, temp=temperature):
                yield char

    @modal.method()
    def prompt(self, prompt: str, temperature: float = 0.1):
        """Predict on the user's output."""
        with self.model.chat_session(), self.mutex:
            return self.model.generate(prompt, temp=temperature)

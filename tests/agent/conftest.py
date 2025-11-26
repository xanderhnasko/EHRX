"""
Test fixtures and stubs for agent tests.

We stub out the Vertex AI SDK so tests can run without external
dependencies or credentials.
"""

import sys
import types
from pathlib import Path

# Ensure project root is importable for `ehrx` modules
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class DummyGenerationConfig:
    """Lightweight stand-in for vertexai.generative_models.GenerationConfig."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class DummyGenerativeModel:
    """Lightweight stand-in for vertexai.generative_models.GenerativeModel."""

    def __init__(self, *args, **kwargs):
        pass

    def generate_content(self, *args, **kwargs):  # pragma: no cover - overridden in tests
        raise NotImplementedError("This is a test stub; override in tests.")


class DummyPart:
    """Lightweight stand-in for vertexai.generative_models.Part."""

    @classmethod
    def from_data(cls, data=None, mime_type=None):
        return cls()


# Build stub module objects
gen_models_stub = types.SimpleNamespace(
    GenerativeModel=DummyGenerativeModel,
    GenerationConfig=DummyGenerationConfig,
    Part=DummyPart
)
vertexai_stub = types.SimpleNamespace(init=lambda *args, **kwargs: None)
setattr(vertexai_stub, "generative_models", gen_models_stub)

# Register stubs so imports inside ehrx.agent.query succeed without the real SDK
sys.modules.setdefault("vertexai", vertexai_stub)
sys.modules.setdefault("vertexai.generative_models", gen_models_stub)

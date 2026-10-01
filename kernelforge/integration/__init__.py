from __future__ import annotations

__all__ = ["BlockConfig", "TransformerBlock", "benchmark_block"]


def __getattr__(name: str):
    """Defer the Triton import until the transformer block is actually used."""
    if name in __all__:
        from kernelforge.integration import transformer

        return getattr(transformer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

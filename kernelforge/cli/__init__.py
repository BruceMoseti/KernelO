"""Command line entry point.

Deliberately empty of imports: ``python -m kernelforge.cli.main`` warns if the
submodule is already in ``sys.modules`` when the package body runs, and the
Nsight integration re-invokes the CLI that way.
"""

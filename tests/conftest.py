"""Shared pytest setup.

Provide a dummy API key so importing app config never fails in CI. Test data
helpers live in tests/factories.py (not here), since importing conftest directly
is discouraged.
"""
import os

os.environ.setdefault("OPENROUTER_API_KEY", "test-key")

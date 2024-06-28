from contextlib import AbstractContextManager as ContextManager
from typing import Protocol

import pytest


class MissingModulesContextGenerator(Protocol):
    def __call__(self, *names: str) -> ContextManager[pytest.MonkeyPatch]: ...

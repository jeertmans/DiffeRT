import builtins
import importlib
import sys
from contextlib import AbstractContextManager as ContextManager
from contextlib import contextmanager
from threading import Lock
from types import ModuleType
from typing import Any

import pytest

from ._types import MissingModulesContextGenerator

_LOCK = Lock()


@pytest.fixture
def missing_modules(monkeypatch: pytest.MonkeyPatch) -> MissingModulesContextGenerator:
    @contextmanager  # type: ignore[arg-type]
    def ctx(*names: str) -> ContextManager[pytest.MonkeyPatch]:  # type: ignore[misc]
        real_import = builtins.__import__
        real_import_module = importlib.import_module

        def monkey_import(name: str, *args: Any, **kwargs: Any) -> ModuleType:
            if name.partition(".")[0] in names:
                raise ImportError(f"Mocked import error for '{name}'")
            return real_import(name, *args, **kwargs)

        def monkey_import_module(name: str, *args: Any, **kwargs: Any) -> ModuleType:
            if name.partition(".")[0] in names:
                raise ImportError(f"Mocked import error for '{name}'")
            return real_import_module(name, *args, **kwargs)

        with monkeypatch.context() as m, _LOCK:
            module_names = tuple(sys.modules.keys())

            for module_name in module_names:
                if module_name.partition(".")[0] in names:
                    m.delitem(sys.modules, module_name)

            m.setattr(builtins, "__import__", monkey_import)
            m.setattr(importlib, "import_module", monkey_import_module)

            yield m  # type: ignore

    return ctx

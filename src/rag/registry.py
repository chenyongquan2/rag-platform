"""Name -> factory registries for the three extension points (design s5-1).

Adding a data source, a file format or a processing step touches only its registry;
knowledge-base YAML refers to entries by name.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class UnknownNameError(LookupError):
    """Raised when a name is not registered; the message lists the registered names."""

    def __str__(self) -> str:  # LookupError would otherwise quote the message like KeyError
        return str(self.args[0]) if self.args else ""


class DuplicateNameError(ValueError):
    """Raised when a name is registered twice in the same registry."""


class Registry[F: Callable[..., Any]]:
    """Maps names to factories (classes or functions)."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: dict[str, F] = {}

    def add(self, name: str, factory: F) -> F:
        if not name:
            raise ValueError(f"{self.kind} name must not be empty")
        if name in self._entries:
            raise DuplicateNameError(f"{self.kind} {name!r} is already registered")
        self._entries[name] = factory
        return factory

    def register(self, name: str) -> Callable[[F], F]:
        """Decorator: `@steps.register("token_split")`."""

        def decorator(factory: F) -> F:
            return self.add(name, factory)

        return decorator

    def get(self, name: str) -> F:
        try:
            return self._entries[name]
        except KeyError:
            known = ", ".join(self.names()) or "(none)"
            raise UnknownNameError(f"unknown {self.kind} {name!r}; registered: {known}") from None

    def create(self, name: str, /, **params: Any) -> Any:
        """Look up `name` and call its factory with `params` (from the KB config)."""
        return self.get(name)(**params)

    def names(self) -> list[str]:
        return sorted(self._entries)

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __repr__(self) -> str:
        return f"Registry({self.kind!r}, names={self.names()!r})"


connectors: Registry[Callable[..., Any]] = Registry("connector")
readers: Registry[Callable[..., Any]] = Registry("reader")
steps: Registry[Callable[..., Any]] = Registry("processing step")

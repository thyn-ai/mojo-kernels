"""Shared exceptions for pymavlink-mojo."""

from __future__ import annotations


class NativeUnavailable(RuntimeError):  # noqa: N818
    """The native pymavmojo kernel could not be found, loaded, or verified."""

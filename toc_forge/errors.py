"""Domain errors exposed by the toc_forge processing interface."""


class TocForgeError(RuntimeError):
    """Base class for expected processing failures."""


class TocNotFoundError(TocForgeError):
    """Raised when no table-of-contents pages can be detected."""


class EmptyTocError(TocForgeError):
    """Raised when extraction produces no resolvable bookmark entries."""

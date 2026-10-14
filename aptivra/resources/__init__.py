"""Access immutable runtime resources distributed with Aptivra."""

from importlib.resources import files
from importlib.resources.abc import Traversable


def require_resource(package: str, *parts: str) -> Traversable:
    """Return a packaged resource or raise an error that names the missing file."""
    resource = files(package).joinpath(*parts)
    if not resource.is_file():
        resource_name = "/".join(parts)
        raise FileNotFoundError(
            f"Required Aptivra packaged resource '{resource_name}' is missing"
        )
    return resource


def aptivra_resource(*parts: str) -> Traversable:
    """Return a resource stored below the ``aptivra.resources`` package."""
    return require_resource(__package__, *parts)

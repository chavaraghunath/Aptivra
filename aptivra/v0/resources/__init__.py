"""Access immutable resources for the historical v0-byte-level namespace."""

from importlib.resources import files
from importlib.resources.abc import Traversable


def aptivra_resource(*parts: str) -> Traversable:
    """Return a resource stored below the historical ``aptivra.v0.resources`` package."""
    resource = files(__package__).joinpath(*parts)
    if not resource.is_file():
        resource_name = "/".join(parts)
        raise FileNotFoundError(
            f"Required Aptivra v0 packaged resource '{resource_name}' is missing"
        )
    return resource

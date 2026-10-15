"""Artifact persistence shared by local rehearsals and Vertex workers."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Protocol, cast


class ArtifactStore(Protocol):
    """Minimal durable-object interface used by the cloud sweep."""

    def exists(self, relative_path: str) -> bool:
        """Return whether an artifact exists."""

    def read_bytes(self, relative_path: str) -> bytes:
        """Read one artifact."""

    def write_bytes(
        self, relative_path: str, payload: bytes, *, overwrite: bool = False
    ) -> None:
        """Write one artifact, rejecting an existing object unless allowed."""

    def relative_paths(self) -> list[str]:
        """List every artifact path relative to the configured root."""


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not value or ".." in path.parts:
        raise ValueError(f"artifact path must be a safe relative path: {value!r}")
    return path.as_posix()


class LocalArtifactStore:
    """Atomic filesystem implementation used by tests and the CPU rehearsal."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _path(self, relative_path: str) -> Path:
        return self.root / _safe_relative_path(relative_path)

    def exists(self, relative_path: str) -> bool:
        return self._path(relative_path).is_file()

    def read_bytes(self, relative_path: str) -> bytes:
        return self._path(relative_path).read_bytes()

    def write_bytes(
        self, relative_path: str, payload: bytes, *, overwrite: bool = False
    ) -> None:
        destination = self._path(relative_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_bytes(payload)
        if destination.exists() and not overwrite:
            temporary.unlink(missing_ok=True)
            raise FileExistsError(destination)
        temporary.replace(destination)

    def relative_paths(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(
            path.relative_to(self.root).as_posix()
            for path in self.root.rglob("*")
            if path.is_file() and not path.name.endswith(".tmp")
        )


class GcsArtifactStore:
    """Small authenticated GCS JSON-API client with bounded retries."""

    _METADATA_TOKEN_URL = (
        "http://metadata.google.internal/computeMetadata/v1/instance/"
        "service-accounts/default/token"
    )

    def __init__(
        self,
        uri: str,
        *,
        timeout_seconds: float,
        retries: int,
        retry_backoff_seconds: float,
        token_provider: Callable[[], str] | None = None,
    ) -> None:
        parsed = urllib.parse.urlsplit(uri)
        if parsed.scheme != "gs" or not parsed.netloc:
            raise ValueError("GCS artifact URI must be gs://BUCKET/PREFIX")
        if timeout_seconds <= 0 or retries < 1 or retry_backoff_seconds < 0:
            raise ValueError("GCS timeout/retry settings are outside safe limits")
        self.bucket = parsed.netloc
        self.prefix = parsed.path.strip("/")
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self._token_provider = token_provider or self._metadata_token

    def _metadata_token(self) -> str:
        request = urllib.request.Request(
            self._METADATA_TOKEN_URL,
            headers={"Metadata-Flavor": "Google"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            raw = response.read()
        value = json.loads(raw)
        if not isinstance(value, dict) or not isinstance(value.get("access_token"), str):
            raise RuntimeError("metadata server returned no access token")
        return cast(str, value["access_token"])

    def _object_name(self, relative_path: str) -> str:
        relative = _safe_relative_path(relative_path)
        return f"{self.prefix}/{relative}" if self.prefix else relative

    def _request(
        self,
        request_factory: Callable[[str], urllib.request.Request],
    ) -> bytes:
        last_error: BaseException | None = None
        for attempt in range(self.retries):
            try:
                request = request_factory(self._token_provider())
                with urllib.request.urlopen(
                    request, timeout=self.timeout_seconds
                ) as response:
                    return cast(bytes, response.read())
            except urllib.error.HTTPError as error:
                last_error = error
                if error.code not in {408, 429, 500, 502, 503, 504}:
                    raise
            except (TimeoutError, urllib.error.URLError) as error:
                last_error = error
            if attempt + 1 < self.retries:
                time.sleep(self.retry_backoff_seconds * (2**attempt))
        raise RuntimeError("GCS request retry budget exhausted") from last_error

    @staticmethod
    def _authorized_request(
        url: str,
        token: str,
        *,
        method: str = "GET",
        data: bytes | None = None,
        content_type: str | None = None,
    ) -> urllib.request.Request:
        headers = {"Authorization": f"Bearer {token}"}
        if content_type is not None:
            headers["Content-Type"] = content_type
        return urllib.request.Request(url, data=data, headers=headers, method=method)

    def exists(self, relative_path: str) -> bool:
        name = urllib.parse.quote(self._object_name(relative_path), safe="")
        url = f"https://storage.googleapis.com/storage/v1/b/{self.bucket}/o/{name}"
        try:
            self._request(lambda token: self._authorized_request(url, token))
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return False
            raise
        return True

    def read_bytes(self, relative_path: str) -> bytes:
        name = urllib.parse.quote(self._object_name(relative_path), safe="")
        url = (
            f"https://storage.googleapis.com/storage/v1/b/{self.bucket}/o/{name}"
            "?alt=media"
        )
        return self._request(lambda token: self._authorized_request(url, token))

    def write_bytes(
        self, relative_path: str, payload: bytes, *, overwrite: bool = False
    ) -> None:
        query = {
            "uploadType": "media",
            "name": self._object_name(relative_path),
        }
        if not overwrite:
            query["ifGenerationMatch"] = "0"
        url = (
            f"https://storage.googleapis.com/upload/storage/v1/b/{self.bucket}/o?"
            + urllib.parse.urlencode(query)
        )
        try:
            self._request(
                lambda token: self._authorized_request(
                    url,
                    token,
                    method="POST",
                    data=payload,
                    content_type="application/octet-stream",
                )
            )
        except urllib.error.HTTPError as error:
            if error.code == 412 and not overwrite:
                raise FileExistsError(relative_path) from error
            raise

    def relative_paths(self) -> list[str]:
        paths: list[str] = []
        page_token: str | None = None
        while True:
            query = {"prefix": f"{self.prefix}/" if self.prefix else ""}
            if page_token is not None:
                query["pageToken"] = page_token
            url = (
                f"https://storage.googleapis.com/storage/v1/b/{self.bucket}/o?"
                + urllib.parse.urlencode(query)
            )

            def request_factory(
                token: str, request_url: str = url
            ) -> urllib.request.Request:
                return self._authorized_request(request_url, token)

            raw = self._request(request_factory)
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise RuntimeError("GCS list response is not an object")
            items = value.get("items", [])
            if not isinstance(items, list):
                raise RuntimeError("GCS list response has malformed items")
            for item in items:
                if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                    raise RuntimeError("GCS list response has malformed object metadata")
                name = cast(str, item["name"])
                relative = name[len(self.prefix) :].lstrip("/") if self.prefix else name
                if relative:
                    paths.append(_safe_relative_path(relative))
            token = value.get("nextPageToken")
            if token is None:
                break
            if not isinstance(token, str):
                raise RuntimeError("GCS list response has malformed page token")
            page_token = token
        return sorted(paths)


def download_store(store: ArtifactStore, destination: Path) -> None:
    """Materialize all durable artifacts into a local evaluator directory."""
    local = LocalArtifactStore(destination)
    for relative_path in store.relative_paths():
        local.write_bytes(relative_path, store.read_bytes(relative_path), overwrite=True)


def upload_file(
    store: ArtifactStore,
    root: Path,
    path: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Upload one local artifact using its root-relative path."""
    relative = path.relative_to(root).as_posix()
    store.write_bytes(relative, path.read_bytes(), overwrite=overwrite)

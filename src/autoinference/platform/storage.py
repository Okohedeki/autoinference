"""Object storage: customer inputs and model outputs.

Bulk data never passes through the control plane. Clients upload to a presigned
URL, workers download inputs and upload outputs by key, and results carry
presigned download URLs. ``LocalStorage`` is a directory with the same
interface, for development and tests; the local "presigned URLs" are served by
the API itself.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Protocol


class Storage(Protocol):
    def put(self, key: str, path: Path, content_type: str | None = None) -> None: ...

    def get(self, key: str, dest: Path) -> Path: ...

    def exists(self, key: str) -> bool: ...

    def delete(self, key: str) -> None: ...

    def upload_url(self, key: str, content_type: str | None = None, expires: int = 3600) -> str: ...

    def download_url(self, key: str, expires: int = 3600) -> str: ...


class LocalStorage:
    def __init__(self, root: Path, public_url: str = "") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.public_url = public_url.rstrip("/")

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if self.root.resolve() not in path.parents:
            raise ValueError(f"key escapes the storage root: {key}")
        return path

    def put(self, key: str, path: Path, content_type: str | None = None) -> None:
        dest = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)

    def put_bytes(self, key: str, data: bytes) -> None:
        dest = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)

    def get(self, key: str, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._path(key), dest)
        return dest

    def read(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def delete(self, key: str) -> None:
        path = self._path(key)
        if path.exists():
            path.unlink()

    def upload_url(self, key: str, content_type: str | None = None, expires: int = 3600) -> str:
        return f"{self.public_url}/v1/files/_local/{key}"

    def download_url(self, key: str, expires: int = 3600) -> str:
        return f"{self.public_url}/v1/files/_local/{key}"


class R2Storage:
    """Cloudflare R2 through its S3-compatible API."""

    def __init__(self, bucket: str, endpoint: str, access_key_id: str, secret_access_key: str) -> None:
        import boto3
        from botocore.config import Config

        self.bucket = bucket
        self.client = boto3.client("s3", endpoint_url=endpoint, aws_access_key_id=access_key_id,
                                   aws_secret_access_key=secret_access_key, region_name="auto",
                                   config=Config(signature_version="s3v4", s3={"addressing_style": "path"}))

    def put(self, key: str, path: Path, content_type: str | None = None) -> None:
        extra = {"ContentType": content_type} if content_type else {}
        self.client.upload_file(str(path), self.bucket, key, ExtraArgs=extra)

    def get(self, key: str, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, key, str(dest))
        return dest

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except self.client.exceptions.ClientError:
            return False

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)

    def upload_url(self, key: str, content_type: str | None = None, expires: int = 3600) -> str:
        params = {"Bucket": self.bucket, "Key": key}
        if content_type:
            params["ContentType"] = content_type
        return self.client.generate_presigned_url("put_object", Params=params, ExpiresIn=expires)

    def download_url(self, key: str, expires: int = 3600) -> str:
        return self.client.generate_presigned_url("get_object", Params={"Bucket": self.bucket, "Key": key},
                                                  ExpiresIn=expires)


def make_storage(settings) -> Storage:
    if settings.storage == "r2":
        return R2Storage(settings.r2_bucket, settings.r2_endpoint, settings.r2_access_key_id,
                         settings.r2_secret_access_key)
    return LocalStorage(settings.storage_dir / "files", settings.public_url)


def input_key(project_id: str, file_id: str) -> str:
    return f"{project_id}/in/{file_id}"


def output_key(project_id: str, request_id: str, name: str) -> str:
    return f"{project_id}/out/{request_id}/{name}"

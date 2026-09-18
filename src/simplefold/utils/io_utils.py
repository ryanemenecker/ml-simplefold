#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import os
import subprocess
from importlib import resources
from pathlib import Path


def get_config_path(relative_path):
    """Get the absolute path to a config file using importlib.resources."""
    try:
        # Remove 'configs/' prefix if present since we access configs directly as a subpackage
        config_subpath = relative_path.replace('configs/', '')

        # Access configs as a subpackage resource
        config_files = resources.files('simplefold.configs')
        config_path = config_files / config_subpath

        if config_path.is_file():
            return str(config_path)

    except Exception as e:
        pass

    # If importlib.resources fails, raise an informative error
    raise FileNotFoundError(
        f"Could not find config file: {relative_path}. "
        f"Expected to find it in the simplefold.configs package."
    )


def download_file(url, path):
    """Download ``url`` to ``path`` with curl, atomically.

    ``-f`` makes HTTP errors fail instead of saving the error page as the checkpoint, ``--retry``
    and ``-C -`` make multi-GB downloads survive transient network drops, and the file is written
    to ``<path>.part`` and renamed only when complete, so an interrupted download can never be
    mistaken for a cached checkpoint on the next run.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")

    def run(resume):
        cmd = ["curl", "-fL", "--retry", "3"] + (["-C", "-"] if resume else []) + ["-o", str(part), url]
        subprocess.run(cmd, check=True)

    try:
        try:
            run(resume=True)
        except subprocess.CalledProcessError as e:
            if e.returncode != 33:  # 33: server does not support byte ranges -> restart from scratch
                raise
            part.unlink(missing_ok=True)
            run(resume=False)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Download of {url} failed (curl exit code {e.returncode}). "
            f"A partial file may remain at {part}; re-run to resume or delete it."
        ) from e
    os.replace(part, path)
    return path

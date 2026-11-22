"""Helpers for accepting legacy environment IDs in command-line tools."""

import re


def normalize_env_id(env_id: str) -> str:
    """Return a namespaced registry ID, accepting legacy names such as ``mobeach_v0``.

    PettingZoo's ``make`` already accepts ``_vN`` suffixes, but tools also compare IDs and derive
    file names from them, so every accepted form is mapped to one canonical ``momaland/<name>-vN`` ID.
    """
    env_id = re.sub(r"_v(\d+)$", r"-v\1", env_id)
    return env_id if "/" in env_id else f"momaland/{env_id}"


def env_id_to_name(env_id: str) -> str:
    """Return a filesystem-safe name such as ``mobeach_v0`` for any accepted environment ID form.

    Used for checkpoint directories, run names and GIF files, so that outputs do not depend on which
    ID form was passed and never contain the namespace separator ``/``.
    """
    name = normalize_env_id(env_id).split("/", 1)[1]
    return re.sub(r"-v(\d+)$", r"_v\1", name)

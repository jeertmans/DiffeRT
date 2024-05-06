"""
Provide a compatibility layer with Sionna's scenes.

Sionna uses the simple XML-based format from Mitsuba 3.
"""

__all__ = (
    "download_sionna_scenes",
    "get_sionna_scene",
    "list_sionna_scenes",
)

import tarfile
import tempfile
from pathlib import Path
from threading import Lock
from typing import Union

import requests
from tqdm import tqdm

SIONNA_SCENES_FOLDER = Path(__file__).parent / "scenes"

_LOCK = Lock()


def download_sionna_scenes(
    branch_or_tag: str = "main",
    *,
    folder: Union[str, Path] = SIONNA_SCENES_FOLDER,
    cached: bool = True,
    chunk_size: int = 1024,
    progress: bool = True,
    leave: bool = False,
) -> None:
    """
    Download the scenes from Sionna, and stores them in the given folder.

    If cached is :py:data:`False` and folder exists, then it will
    raise an error if not empty: please clear it first!

    Args:
        branch_or_tag: The branch or tag version of the Sionna repository.
        folder: Where to extract the scene files, i.e., the content
            of ``sionna/rt/scenes/``.
        cached: Whether to avoid downloading again if the target folder
            already exists.
        chunk_size: The chunk size, in bytes, used when downloading
            the data.
        progress: Whether to output a progress bar when downloading.
        leave: If ``progress`` is :py:data:`True`, whether to leave
            the progress bar upon completion.
    """
    if isinstance(folder, str):
        folder = Path(folder)

    with _LOCK:
        if folder.exists():
            if cached:
                return

            folder.rmdir()

        url = f"https://codeload.github.com/NVlabs/sionna/tar.gz/{branch_or_tag}"

        response = requests.get(url, stream=True)

        stream = response.iter_content(chunk_size=chunk_size)
        total = int(response.headers.get("content-length", 0))

        def members(tar: tarfile.TarFile):
            for member in tar.getmembers():
                if (index := member.path.find("sionna/rt/scenes/")) >= 0:
                    member.path = member.path[index + 17 :]
                    yield member

        with (
            tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f,
            tqdm(
                stream,
                desc="Downloading Sionna's repository archive...",
                total=total,
                unit="iB",
                unit_scale=True,
                unit_divisor=chunk_size,
                disable=not progress,
                leave=leave,
            ) as bar,
        ):
            for chunk in stream:
                size = f.write(chunk)
                bar.update(size)

            f.flush()

            with tarfile.open(f.name) as tar:
                tar.extractall(path=folder, members=members(tar), filter="data")


def list_sionna_scenes(*, folder: Union[str, Path] = SIONNA_SCENES_FOLDER) -> list[str]:
    """
    List available Sionna scenes, by name.

    Args:
        folder: Where scene files are stored.

    Return:
        The list of scene names.
    """
    if isinstance(folder, str):
        folder = Path(folder)

    return [p.name for p in folder.iterdir() if p.is_dir() and p.name != "__pycache__"]


def get_sionna_scene(
    scene_name: str, *, folder: Union[str, Path] = SIONNA_SCENES_FOLDER
) -> str:
    """
    Return the path to the given Sionna scene.

    Args:
        scene_name: The name of the scene.
        folder: Where scene files are stored.

    Return:
        The path, relative to the current working directory,
        to the given ``scene.xml`` file.
    """
    if isinstance(folder, str):
        folder = Path(folder)

    p = folder / scene_name / f"{scene_name}.xml"

    if not p.exists():
        scenes = ", ".join(list_sionna_scenes(folder=folder))
        raise ValueError(
            f"Cannot find {scene_name = }! Available scenes are: {scenes}."
        )

    return str(p)

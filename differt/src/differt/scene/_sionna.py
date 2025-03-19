# /// script
# dependencies = [
#   "filelock>=3.15.4",
#   "requests>=2.32.0",
#   "tqdm>=4.66.2",
# ]
# ///

import sys
import tarfile
import tempfile
import warnings
from collections.abc import Iterator
from pathlib import Path

import requests
from filelock import FileLock
from tqdm.auto import tqdm

SIONNA_SCENES_FOLDER = Path(__file__).parent / "scenes"


def download_sionna_scenes(
    branch_or_tag: str = "main",
    repo: str = "NVlabs/sionna-rt",
    scenes_folder: str = "src/sionna/rt/scenes/",
    *,
    folder: str | Path = SIONNA_SCENES_FOLDER,
    cached: bool = True,
    chunk_size: int = 1024,
    progress: bool = True,
    leave: bool = False,
    timeout: float | tuple[float, float] | None = None,
) -> None:
    """
    Download the scenes from Sionna, and store them in the given folder.

    If cached is :data:`False` and folder exists, then it will
    raise an error if not empty: please clear it first!

    Warning:
        Older Python versions (i.e., <3.12, <3.11.4, and <3.10.12) do not
        provide the ``filter`` parameter in :meth:`tarfile.TarFile.extractall`,
        which can be considered a security risk. This function will raise a
        warning if the current Python version is one of these versions.

    Args:
        branch_or_tag: The branch or tag version of the Sionna repository.
        repo: The repository ``{owner}/{name}`` on GitHub from which to download the scenes.

            E.g., you can change this if you want to use a fork instead.
        scenes_folder: The folder where the scenes are stored in the
            repository.
        folder: Where to extract the scene files, i.e., the content
            of ``sionna/rt/scenes/``.
        cached: Whether to avoid downloading again if the target folder
            already exists.
        chunk_size: The chunk size, in bytes, used when downloading
            the data.
        progress: Whether to output a progress bar when downloading.
        leave: If ``progress`` is :data:`True`, whether to leave
            the progress bar upon completion.
        timeout: How many seconds to wait before giving up on the download,
            see :func:`requests.request`.
    """
    if isinstance(folder, str):
        folder = Path(folder)

    with FileLock(folder.with_name("scenes.lock")):
        if folder.exists():
            if cached:
                return

            folder.rmdir()

        url = f"https://codeload.github.com/{repo}/tar.gz/{branch_or_tag}"

        response = requests.get(url, stream=True, timeout=timeout)

        stream = response.iter_content(chunk_size=chunk_size)
        total = int(response.headers.get("content-length", 0))

        def members(tar: tarfile.TarFile) -> Iterator[tarfile.TarInfo]:
            for member in tar.getmembers():
                if (index := member.path.find(scenes_folder)) >= 0:
                    member.path = member.path[index + len(scenes_folder) :]
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
                # tarfile added 'filter' parameter for security reasons.
                if (
                    sys.version_info >= (3, 12)
                    or (sys.version_info.minor == 11 and sys.version_info.micro >= 4)  # noqa: PLR2004
                    or (sys.version_info.minor == 10 and sys.version_info.micro >= 12)  # noqa: PLR2004
                ):
                    tar.extractall(path=folder, members=members(tar), filter="data")
                else:  # pragma: no cover
                    msg = (
                        "You are using an old version of Python that doesn't include the 'filter' "
                        "parameter in 'tarfile.TarFile.extractall'. This is can be security issue, and we "
                        "recommend upgrading to a newer version of Python: 3.12, 3.11.4, or 3.10.12."
                    )
                    warnings.warn(msg, UserWarning, stacklevel=2)
                    tar.extractall(path=folder, members=members(tar))  # noqa: S202


def list_sionna_scenes(*, folder: str | Path = SIONNA_SCENES_FOLDER) -> list[str]:
    """
    List available Sionna scenes, by name.

    Args:
        folder: Where scene files are stored.

    Returns:
        The list of scene names.
    """
    if isinstance(folder, str):
        folder = Path(folder)

    return [p.name for p in folder.iterdir() if p.is_dir() and p.name != "__pycache__"]


def get_sionna_scene(
    scene_name: str,
    *,
    folder: str | Path = SIONNA_SCENES_FOLDER,
) -> str:
    """
    Return the path to the given Sionna scene.

    Args:
        scene_name: The name of the scene.
        folder: Where scene files are stored.

    Returns:
        The path, relative to the current working directory,
        to the given ``scene.xml`` file.

    Raises:
        ValueError: If scene does not exist.
    """
    if isinstance(folder, str):
        folder = Path(folder)

    p = folder / scene_name / f"{scene_name}.xml"

    if not p.exists():
        scenes = ", ".join(list_sionna_scenes(folder=folder))
        msg = f"Cannot find {scene_name = }! Available scenes are: {scenes}."
        raise ValueError(
            msg,
        )

    return str(p)


def main() -> None:  # pragma: no cover
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        prog="download-sionna-scenes",
        description="Download Sionna scenes from Sionna's repository.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "ref",
        type=str,
        default="main",
        nargs="?",
        help="branch or tag version of the Sionna repository.",
    )
    parser.add_argument(
        "-r",
        "--repo",
        type=str,
        default="NVlabs/sionna-rt",
        help="the repository '{owner}/{name}' on GitHub from which to download the scenes.",
    )
    parser.add_argument(
        "-s",
        "--scenes-folder",
        type=Path,
        default="src/sionna/rt/scenes",
        help="the folder where the scenes are stored in the repository.",
    )
    parser.add_argument(
        "--folder",
        type=Path,
        default=SIONNA_SCENES_FOLDER,
        help="where to extract the scene files",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="force to clean any existing folder with '--no-cached' is set.",
    )
    parser.add_argument(
        "--cached",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="whether to skip download if folder already exists.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help="the chunk size, in bytes, used when for displaying progress",
    )
    parser.add_argument(
        "--progress",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="whether to show a progress bar",
    )
    parser.add_argument(
        "--leave",
        action="store_true",
        help="whether to leave the progress bar upon completion",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="how many seconds to wait before giving up on the download",
    )
    args = parser.parse_args()

    if args.folder.exists():
        if not args.folder.is_dir():
            msg = f"'{args.folder.relative_to(Path.cwd())}' exists and is not a directory."
            parser.error(msg)
        if not args.cached and len(list(args.folder.iterdir())) > 0:
            if not args.force:
                msg = f"'{args.folder.relative_to(Path.cwd())}' exists and is not empty, please clean it manually or set '--force'."
                parser.error(msg)

            import shutil  # noqa: PLC0415

            shutil.rmtree(args.folder)

    download_sionna_scenes(
        branch_or_tag=args.ref,
        folder=args.folder,
        cached=args.cached,
        chunk_size=args.chunk_size,
        progress=args.progress,
        leave=args.leave,
        timeout=args.timeout,
    )


if __name__ == "__main__":  # pragma: no cover
    main()

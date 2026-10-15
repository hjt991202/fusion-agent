"""Copy only known checkpoints from the owner's archives into ignored local storage."""

import argparse
import hashlib
from pathlib import Path
from zipfile import ZipFile

CHECKPOINTS = {
    "icig": (
        "ICIGCode/weights/model.pth",
        "model.pth",
        "6f32ba4b019faa388e6ddf58b3e3c83c16183dea319d3654b27a04a3262f87fc",
    ),
    "letters": (
        "letters/weights/fusion.pt",
        "fusion.pt",
        "16e61b98ff9f2c95a97f553ebd402f2b93faa37373621f494d6d92fa264a8fc0",
    ),
}


def import_checkpoint(archive_path: Path, method: str, destination: Path) -> Path:
    member, filename, expected_hash = CHECKPOINTS[method]
    with ZipFile(archive_path) as archive:
        info = archive.getinfo(member)
        if info.file_size > 16 * 1024 * 1024:
            raise ValueError("Checkpoint exceeds the known archive size limit.")
        data = archive.read(member)
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError("Checkpoint does not match the reviewed research archive.")
    target = destination / method / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected_hash:
            raise FileExistsError("Refusing to overwrite a different checkpoint.")
    else:
        target.write_bytes(data)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--icig", type=Path)
    parser.add_argument("--letters", type=Path)
    parser.add_argument("--destination", type=Path, default=Path("weights"))
    args = parser.parse_args()
    if not args.icig and not args.letters:
        parser.error("Supply --icig ICIGCode.zip and/or --letters lettersCode.zip.")
    for method in CHECKPOINTS:
        archive = getattr(args, method)
        if archive:
            print(import_checkpoint(archive, method, args.destination))


if __name__ == "__main__":
    main()

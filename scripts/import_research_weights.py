"""Copy only known checkpoints from the owner's archives into ignored local storage."""

import argparse
import hashlib
import io
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
    "prcv": (
        "PRCV_Fusion_submit/weights/model.pth",
        "model.pth",
        "eb430bf79a58a5c9987e6d2ce6dddf54c5fea7e281a78e7cdcd95b721982551d",
    ),
}


def import_checkpoint(archive_path: Path, method: str, destination: Path) -> Path:
    member, filename, expected_hash = CHECKPOINTS[method]
    with ZipFile(archive_path) as archive:
        if method == "prcv" and member not in archive.namelist():
            nested = "JDFusion/PRCV_Fusion_submit.zip"
            if archive.getinfo(nested).file_size > 32 * 1024 * 1024:
                raise ValueError("Nested PRCV archive exceeds the reviewed size limit.")
            with ZipFile(io.BytesIO(archive.read(nested))) as inner:
                data = read_checkpoint(inner, member)
        else:
            data = read_checkpoint(archive, member)
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


def read_checkpoint(archive: ZipFile, member: str) -> bytes:
    """Read one exact member, bounded before allocating its uncompressed contents."""
    if archive.getinfo(member).file_size > 16 * 1024 * 1024:
        raise ValueError("Checkpoint exceeds the known archive size limit.")
    return archive.read(member)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--icig", type=Path)
    parser.add_argument("--letters", type=Path)
    parser.add_argument(
        "--prcv", type=Path, help="PRCV_Fusion_submit.zip or its JDFusion.zip bundle"
    )
    parser.add_argument("--destination", type=Path, default=Path("weights"))
    args = parser.parse_args()
    if not any(getattr(args, method) for method in CHECKPOINTS):
        parser.error(
            "Supply --icig ICIGCode.zip, --letters lettersCode.zip or --prcv JDFusion.zip."
        )
    for method in CHECKPOINTS:
        archive = getattr(args, method)
        if archive:
            print(import_checkpoint(archive, method, args.destination))


if __name__ == "__main__":
    main()

import argparse, pathlib, zipfile, gdown

CITYSCAPES_ID = "1OlElYRhKovWEc8gu32E8at-xNfEQniGq"
WEIGHTS_ID = "1VOmwEwd73ktbCaSlugCPRGEwq9QY85aW"

def gdrive(id_: str, out: pathlib.Path):
    url = f"https://drive.google.com/uc?id={id_}"
    gdown.download(url, str(out), quiet=False)

def main(dest: str) -> None:
    dest = pathlib.Path(dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)

    zip_path = dest / "Cityscapes.zip"
    gdrive(CITYSCAPES_ID, zip_path)
    with zipfile.ZipFile(zip_path) as zf: zf.extractall(dest)
    zip_path.unlink()

    weights_path = "models/deeplabv2/deeplabv2_cityscapes.pt"
    if not weights_path.exists():
        gdrive(WEIGHTS_ID, weights_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args.dest)

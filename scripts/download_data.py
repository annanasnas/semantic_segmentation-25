import argparse, pathlib, zipfile, gdown

FILE_ID = "1OlElYRhKovWEc8gu32E8at-xNfEQniGq"
URL     = f"https://drive.google.com/uc?id={FILE_ID}"

def main(dest: str) -> None:
    dest = pathlib.Path(dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)

    zip_path = dest / "Cityscapes.zip"
    gdown.download(URL, str(zip_path), quiet=False)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    zip_path.unlink()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default="datasets/data")
    args = parser.parse_args()
    main(args.dest)
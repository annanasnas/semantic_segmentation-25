import pathlib, zipfile, gdown

CITYSCAPES_ID = "1OlElYRhKovWEc8gu32E8at-xNfEQniGq"
WEIGHTS_ID = "1VOmwEwd73ktbCaSlugCPRGEwq9QY85aW"

def gdrive(id_: str, out_path: pathlib.Path):
    url = f"https://drive.google.com/uc?id={id_}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(url, str(out_path), quiet=False)

def main():
    
    zip_path = pathlib.Path("datasets/data/Cityscapes.zip")
    gdrive(CITYSCAPES_ID, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(zip_path.parent)
    zip_path.unlink()

    weights_path = pathlib.Path("models/deeplabv2/DeepLab_resnet_pretrained_imagenet.pt")
    if not weights_path.exists():
        gdrive(WEIGHTS_ID, weights_path)

if __name__ == "__main__":
    main()

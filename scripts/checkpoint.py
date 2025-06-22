from pathlib import Path
import torch, os

class Checkpoint:
    def __init__(self, root="checkpoints"):
        self.dir  = Path(root)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.best = -float("inf")

    @staticmethod
    def _atomic_save(obj: dict, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        torch.save(obj, tmp)
        os.replace(tmp, path)

    def save_best(self, metric: float, payload: dict) -> None:
        if metric > self.best:
            self.best = metric
            self._atomic_save(payload, self.dir / "best.pt")
            print(f"New best mIoU = {metric:.4f}")

    def save_final(self, payload: dict, epoch: int) -> None:
        name = f"final_e{epoch}.pt"
        self._atomic_save(payload, self.dir / name)
        print(f"Final checkpoint saved {name}")

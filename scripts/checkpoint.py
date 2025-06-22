from pathlib import Path
import torch, os

class Checkpoint:
    def __init__(self, root="checkpoints"):
        self.dir = Path(root)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.best = -float("inf")

    @staticmethod
    def _atomic_save(obj, path: Path):
        tmp = path.with_suffix(".tmp")
        torch.save(obj, tmp)
        os.replace(tmp, path)

    def save_best(self, metric, payload):
        if metric > self.best:
            self.best = metric
            self._atomic_save(payload, self.dir / "best.pth")
            print(f"Best mIoU = {metric:.4f}")

    def save_final(self, payload, epoch):
        name = self.dir / f"final_e{epoch}.pth"
        self._atomic_save(payload, name)

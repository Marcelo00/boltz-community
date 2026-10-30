from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities import rank_zero_info


class TrainOnly(Callback):
    """Train only the given top-level modules; freeze the rest in eval mode."""

    def __init__(self, trainable: list[str]) -> None:
        """Initialize the callback.

        Parameters
        ----------
        trainable : list[str]
            The names of the top-level modules to train.

        """
        self.trainable = set(trainable)

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:  # noqa: ARG002
        """Freeze the other modules, before the optimizer is created."""
        children = dict(pl_module.named_children())
        unknown = self.trainable - children.keys()
        assert not unknown, f"Unknown modules to train: {sorted(unknown)}"
        assert not any("." not in n for n, _ in pl_module.named_parameters()), (
            "parameters registered outside child modules"
        )

        for name, module in children.items():
            module.requires_grad_(name in self.trainable)
            if name not in self.trainable:
                # Stays in eval mode: Lightning restores each submodule's mode after validation
                module.eval()

        n_train = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in pl_module.parameters())
        assert n_train > 0, "nothing to train"
        rank_zero_info(
            f"Training {n_train:,} of {n_total:,} parameters: {sorted(self.trainable)}"
        )

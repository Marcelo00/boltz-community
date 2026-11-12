import torch
from torch import Tensor, nn
from torchmetrics import MeanMetric

from boltz.data import const
from boltz.model.loss.bfactor import bfactor_loss_fn
from boltz.model.loss.distogramv2 import distogram_loss
from boltz.model.loss.validation import factored_lddt_loss, weighted_minimum_rmsd

_METRICS = [
    "lddt",
    "lddt_best",
    "rmsd",
    "rmsd_best",
    "disto_loss",
    "bfactor_loss",
    *(f"lddt_{name}{suffix}" for name in const.out_types for suffix in ("", "_best")),
]


class MDValidator(nn.Module):
    """Score sampled structures against the first frame of the MD ensemble."""

    def __init__(self, val_names: list[str]) -> None:
        """Initialize the validator.

        Parameters
        ----------
        val_names : list[str]
            The labels of the validation groups to score.

        """
        super().__init__()
        self.val_names = val_names
        self.metrics = nn.ModuleDict({name: MeanMetric() for name in _METRICS})

    def run_model(self, model: nn.Module, batch: dict[str, Tensor], idx_dataset: int):
        """Sample structures for the batch with the validation settings.

        Parameters
        ----------
        model : nn.Module
            The model to evaluate.
        batch : dict[str, Tensor]
            The input features.
        idx_dataset : int
            The index of the validation group.

        Returns
        -------
        dict[str, Tensor]
            The model output.

        """
        args = model.validation_args
        return model(
            batch,
            recycling_steps=args.recycling_steps,
            num_sampling_steps=args.sampling_steps,
            diffusion_samples=args.diffusion_samples,
        )

    def process(
        self,
        model: nn.Module,
        batch: dict[str, Tensor],
        out: dict[str, Tensor],
        idx_dataset: int,  # noqa: ARG002
    ) -> None:
        """Accumulate the lDDT and RMSD of the sampled structures and the head losses.

        Parameters
        ----------
        model : nn.Module
            The evaluated model.
        batch : dict[str, Tensor]
            The input features.
        out : dict[str, Tensor]
            The model output.
        idx_dataset : int
            The index of the validation group.

        """
        samples = out["sample_atom_coords"]
        num_samples = samples.shape[0]
        true_coords = batch["coords"][:, 0].repeat_interleave(num_samples, 0)
        atom_mask = batch["atom_resolved_mask"].repeat_interleave(num_samples, 0)

        lddt, total = factored_lddt_loss(
            true_coords, samples, batch, atom_mask, multiplicity=num_samples
        )
        for name, value in lddt.items():
            weight = total[name][0]
            if weight > 0:
                self._update(f"lddt_{name}", value.mean(), weight)
                self._update(f"lddt_{name}_best", value.max(), weight)

        # Pair-weighted lDDT over all modalities, one value per sample
        pairs = torch.stack([total[name] for name in lddt]).sum(0)
        overall = sum(lddt[name] * total[name] for name in lddt) / pairs.clamp(min=1)
        self._update("lddt", overall.mean())
        self._update("lddt_best", overall.max())

        rmsd, best_rmsd = weighted_minimum_rmsd(samples, batch, multiplicity=num_samples)
        self._update("rmsd", rmsd.mean())
        self._update("rmsd_best", best_rmsd.mean())

        self._process_heads(model, batch, out)

    def _process_heads(
        self, model: nn.Module, batch: dict[str, Tensor], out: dict[str, Tensor]
    ) -> None:
        """Accumulate the losses of the distogram and bfactor heads."""
        disto_loss, _ = distogram_loss(
            out, batch, aggregate_distogram=model.aggregate_distogram
        )
        self._update("disto_loss", disto_loss)
        if model.predict_bfactor:
            self._update(
                "bfactor_loss",
                bfactor_loss_fn(out, batch, model.bfactor_module.boundaries),
            )

    def on_epoch_end(self, model: nn.Module) -> None:
        """Log and reset the accumulated metrics.

        Parameters
        ----------
        model : nn.Module
            The model used for logging.

        """
        for name, metric in self.metrics.items():
            if metric.update_called:
                model.log(f"val/{name}", metric.compute(), prog_bar=name == "lddt")
                metric.reset()

    def _update(self, name: str, value: Tensor, weight: Tensor = 1.0) -> None:
        """Update a metric."""
        self.metrics[name].update(value, weight)

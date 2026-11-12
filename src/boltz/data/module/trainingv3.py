from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from rdkit.Chem import Mol
from torch import Tensor
from torch.utils.data import DataLoader

from boltz.data.crop.cropper import Cropper
from boltz.data.feature.featurizerv2 import Boltz2Featurizer
from boltz.data.filter.dynamic.filter import DynamicFilter
from boltz.data.module.inferencev2 import collate
from boltz.data.mol import load_canonicals, load_molecules
from boltz.data.sample.sampler import Sample, Sampler
from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
from boltz.data.types import Input, Manifest, Record, StructureV2

_MAX_GETITEM_RETRIES = 10
_MAX_SEQS = 1
_NUCLEOTIDES = ["A", "G", "C", "U", "DA", "DG", "DC", "DT"]


@dataclass
class DatasetConfig:
    """Dataset configuration."""

    target_dir: str
    prob: float
    sampler: Sampler
    cropper: Cropper
    filters: Optional[list] = None
    split: Optional[str] = None
    manifest_path: Optional[str] = None


@dataclass
class DataConfig:
    """Data configuration."""

    datasets: list[DatasetConfig]
    filters: list[DynamicFilter]
    mol_dir: str
    max_atoms: int
    max_tokens: int
    samples_per_epoch: int
    batch_size: int
    num_workers: int
    pin_memory: bool
    atoms_per_window_queries: int = 32
    min_dist: float = 2.0
    max_dist: float = 22.0
    num_bins: int = 64
    num_ensembles: int = 1
    ensemble_sample_replacement: bool = True
    fix_single_ensemble: bool = False
    disto_use_ensemble: bool = False
    override_method: Optional[str] = "md"
    bfactor_md_correction: bool = True
    val_name: str = "md_val"
    num_val_records: Optional[int] = None
    val_seed: int = 0
    overfit: Optional[int] = None


@dataclass
class Dataset:
    """Data holder."""

    target_dir: Path
    manifest: Manifest
    prob: float
    sampler: Sampler
    cropper: Cropper


def load_input(record: Record, target_dir: Path) -> Input:
    """Load the given input data.

    Parameters
    ----------
    record : Record
        The record to load.
    target_dir : Path
        The path to the data directory.

    Returns
    -------
    Input
        The loaded input, without MSAs since the data has no proteins.

    """
    structure = StructureV2.load(target_dir / "structures" / f"{record.id}.npz")
    return Input(structure, {}, record=record)


class BaseDataset(torch.utils.data.Dataset):
    """Base dataset turning a sampled record into model features."""

    def __init__(self, datasets: list[Dataset], cfg: DataConfig) -> None:
        """Initialize the dataset.

        Parameters
        ----------
        datasets : list[Dataset]
            The datasets to load from.
        cfg : DataConfig
            The data configuration.

        """
        super().__init__()
        self.datasets = datasets
        self.cfg = cfg
        self.tokenizer = Boltz2Tokenizer()
        self.featurizer = Boltz2Featurizer()
        self.canonicals = load_canonicals(cfg.mol_dir)
        self._mol_cache = load_molecules(cfg.mol_dir, _NUCLEOTIDES)

    def _featurize(
        self,
        dataset: Dataset,
        sample: Sample,
        random: np.random.Generator,
        crop_random: np.random.RandomState,
        training: bool,
        **ensemble_options,
    ) -> dict[str, Tensor]:
        """Load, tokenize, crop and featurize a sampled record."""
        input_data = load_input(sample.record, dataset.target_dir)
        tokenized = self.tokenizer.tokenize(input_data)
        tokenized = dataset.cropper.crop(
            tokenized,
            max_atoms=self.cfg.max_atoms,
            max_tokens=self.cfg.max_tokens,
            random=crop_random,
            chain_id=sample.chain_id,
            interface_id=sample.interface_id,
        )
        if len(tokenized.tokens) == 0:
            msg = "No tokens in cropped structure"
            raise ValueError(msg)

        return self.featurizer.process(
            tokenized,
            random=random,
            molecules=self._get_molecules(tokenized.tokens),
            training=training,
            max_seqs=_MAX_SEQS,
            atoms_per_window_queries=self.cfg.atoms_per_window_queries,
            min_dist=self.cfg.min_dist,
            max_dist=self.cfg.max_dist,
            num_bins=self.cfg.num_bins,
            override_method=self.cfg.override_method,
            bfactor_md_correction=self.cfg.bfactor_md_correction,
            **ensemble_options,
        )

    def _get_molecules(self, tokens: np.ndarray) -> dict[str, Mol]:
        """Get the reference molecules of all residues in the tokens."""
        molecules = dict(self.canonicals)
        names = set(tokens["res_name"].tolist()) - set(molecules)
        uncached = names - set(self._mol_cache)
        if uncached:
            self._mol_cache.update(load_molecules(self.cfg.mol_dir, uncached))
        molecules.update({name: self._mol_cache[name] for name in names})
        return molecules


class TrainingDataset(BaseDataset):
    """Dataset sampling records with replacement, with random MD frames."""

    def __init__(self, datasets: list[Dataset], cfg: DataConfig) -> None:
        """Initialize the training dataset.

        Parameters
        ----------
        datasets : list[Dataset]
            The datasets to sample from.
        cfg : DataConfig
            The data configuration.

        """
        super().__init__(datasets, cfg)
        self.probs = [d.prob for d in datasets]
        self._random: Optional[np.random.Generator] = None
        self.samples = []
        for dataset in datasets:
            records = dataset.manifest.records
            if cfg.overfit is not None:
                records = records[: cfg.overfit]
            self.samples.append(dataset.sampler.sample(records, np.random))

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Get an item from the dataset.

        Parameters
        ----------
        idx : int
            The data index.

        Returns
        -------
        dict[str, Tensor]
            The sampled data features.

        """
        for _retry in range(_MAX_GETITEM_RETRIES):
            dataset_idx = np.random.choice(len(self.datasets), p=self.probs)
            sample: Sample = next(self.samples[dataset_idx])
            try:
                return self._featurize(
                    self.datasets[dataset_idx],
                    sample,
                    random=self._get_random(),
                    crop_random=np.random,
                    training=True,
                    num_ensembles=self.cfg.num_ensembles,
                    ensemble_sample_replacement=self.cfg.ensemble_sample_replacement,
                    fix_single_ensemble=self.cfg.fix_single_ensemble,
                    disto_use_ensemble=self.cfg.disto_use_ensemble,
                )
            except Exception as e:  # noqa: BLE001
                print(f"Failed to process {sample.record.id} with error {e}. Skipping.")

        msg = f"Failed to load any sample after {_MAX_GETITEM_RETRIES} retries"
        raise RuntimeError(msg)

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.

        """
        return self.cfg.samples_per_epoch

    def _get_random(self) -> np.random.Generator:
        """Get the featurizer generator, created on first use in each worker."""
        # A generator built in __init__ would be copied with the same state into every worker
        if self._random is None:
            self._random = np.random.default_rng(torch.initial_seed() % 2**32)
        return self._random


def select_validation_records(records: list[Record], cfg: DataConfig) -> list[Record]:
    """Select the records to validate on.

    With ``overfit`` these are the first records, the ones training samples from.
    Otherwise ``num_val_records`` records are drawn at random with ``val_seed``.
    """
    if cfg.overfit is not None:
        return records[: cfg.overfit]
    if cfg.num_val_records is None or cfg.num_val_records >= len(records):
        return records
    rng = np.random.RandomState(cfg.val_seed)
    keep = sorted(rng.choice(len(records), cfg.num_val_records, replace=False))
    return [records[i] for i in keep]


class ValidationDataset(BaseDataset):
    """Dataset over the selected records of each dataset, scored against frame 0.

    With ``overfit`` the datasets are the training ones, so validation measures how
    well the training records are fit.
    """

    def __init__(self, datasets: list[Dataset], cfg: DataConfig) -> None:
        """Initialize the validation dataset.

        Parameters
        ----------
        datasets : list[Dataset]
            The datasets to take the validation records from.
        cfg : DataConfig
            The data configuration.

        """
        super().__init__(datasets, cfg)
        self.records = [
            (dataset, record)
            for dataset in datasets
            for record in select_validation_records(dataset.manifest.records, cfg)
        ]

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Get an item from the dataset.

        Parameters
        ----------
        idx : int
            The data index.

        Returns
        -------
        dict[str, Tensor]
            The data features, with the validation group index.

        """
        dataset, record = self.records[idx]
        features = self._featurize(
            dataset,
            Sample(record=record),
            random=np.random.default_rng(idx),
            crop_random=np.random.RandomState(idx),
            training=False,
            num_ensembles=1,
            fix_single_ensemble=True,
            disto_use_ensemble=self.cfg.disto_use_ensemble,
        )
        features["idx_dataset"] = torch.tensor(0)
        return features

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.

        """
        return len(self.records)


class BoltzTrainingDataModule(pl.LightningDataModule):
    """DataModule for boltz2 training."""

    def __init__(self, cfg: DataConfig) -> None:
        """Initialize the DataModule.

        Parameters
        ----------
        config : DataConfig
            The data configuration.

        """
        super().__init__()
        self.cfg = cfg

        train: list[Dataset] = []
        val: list[Dataset] = []
        for data_config in cfg.datasets:
            records = self._load_records(data_config)
            train_records, val_records = self._split_records(records, data_config.split)
            train_records = self._filter_records(train_records, data_config)
            train.append(self._create_dataset(data_config, train_records))
            val.append(self._create_dataset(data_config, val_records))
            print(  # noqa: T201
                f"Training dataset size: {len(train_records)}, "
                f"validation: {len(val_records)}"
            )

        self._train_set = TrainingDataset(train, cfg)
        self._val_set = ValidationDataset(train if cfg.overfit is not None else val, cfg)
        self.val_group_mapper = {0: {"label": cfg.val_name}}

    @staticmethod
    def _load_records(data_config: DatasetConfig) -> list[Record]:
        """Load the manifest records of a dataset."""
        target_dir = Path(data_config.target_dir)
        path = (
            Path(data_config.manifest_path)
            if data_config.manifest_path is not None
            else target_dir / "manifest.json"
        )
        return Manifest.load(path).records

    def _filter_records(
        self, records: list[Record], data_config: DatasetConfig
    ) -> list[Record]:
        """Keep the records that pass the dynamic filters; only training is filtered."""
        filters = [*self.cfg.filters, *(data_config.filters or [])]
        return [r for r in records if all(f.filter(r) for f in filters)]

    @staticmethod
    def _split_records(
        records: list[Record], split: Optional[str]
    ) -> tuple[list[Record], list[Record]]:
        """Split the records in training and validation by the ids in the split file."""
        val_ids = set()
        if split is not None:
            with Path(split).open("r") as f:
                val_ids = {x.lower() for x in f.read().splitlines()}
        train_records = [r for r in records if r.id.lower() not in val_ids]
        val_records = [r for r in records if r.id.lower() in val_ids]
        return train_records, val_records

    @staticmethod
    def _create_dataset(data_config: DatasetConfig, records: list[Record]) -> Dataset:
        """Create a dataset over the given records."""
        return Dataset(
            Path(data_config.target_dir),
            Manifest(records),
            data_config.prob,
            data_config.sampler,
            data_config.cropper,
        )

    def train_dataloader(self) -> DataLoader:
        """Get the training dataloader.

        Returns
        -------
        DataLoader
            The training dataloader.

        """
        return DataLoader(
            self._train_set,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            shuffle=False,
            collate_fn=collate,
        )

    def val_dataloader(self) -> DataLoader:
        """Get the validation dataloader.

        Returns
        -------
        DataLoader
            The validation dataloader.

        """
        # validation_step supports a batch size of 1 only (because of ensemble)
        return DataLoader(
            self._val_set,
            batch_size=1,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            shuffle=False,
            collate_fn=collate,
        )

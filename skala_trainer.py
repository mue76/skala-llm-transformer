"""
Educational trainer adapted from Andrej Karpathy's minGPT.
Original project: https://github.com/karpathy/minGPT
License: MIT

The training loop and callback interface follow minGPT's Trainer design.
CfgNode dependency was replaced with a standard dataclass config.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import torch
from torch.utils.data import DataLoader, RandomSampler


@dataclass
class TrainerConfig:
    device: str = "auto"
    num_workers: int = 0
    max_iters: Optional[int] = None
    batch_size: int = 64
    learning_rate: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    grad_norm_clip: float = 1.0


class Trainer:
    """A lightweight PyTorch trainer preserving minGPT's core training flow."""

    def __init__(self, config: TrainerConfig, model, train_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks: Dict[str, list[Callable[[Any], None]]] = {}

        if config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.device

        self.model = self.model.to(self.device)
        print("running on device", self.device)

        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0
        self.loss = None

    def add_callback(self, onevent: str, callback: Callable[[Any], None]) -> None:
        self.callbacks.setdefault(onevent, []).append(callback)

    def set_callback(self, onevent: str, callback: Callable[[Any], None]) -> None:
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str) -> None:
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self) -> None:
        model, config = self.model, self.config

        self.optimizer = model.configure_optimizers(config)

        train_loader = DataLoader(
            self.train_dataset,
            sampler=RandomSampler(
                self.train_dataset,
                replacement=True,
                num_samples=int(1e10),
            ),
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)

        while True:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            batch = [item.to(self.device) for item in batch]
            x, y = batch

            _, self.loss = model(x, y)

            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_norm_clip
            )
            self.optimizer.step()

            self.trigger_callbacks("on_batch_end")

            self.iter_num += 1
            now = time.time()
            self.iter_dt = now - self.iter_time
            self.iter_time = now

            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break

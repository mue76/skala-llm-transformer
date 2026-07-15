"""학습 루프(Trainer) 모듈.

Andrej Karpathy의 minGPT(MIT License) trainer를 핵심 로직 변경 없이
교육용으로 재구성한 버전입니다.
- 원본: https://github.com/karpathy/minGPT
- 변경점: CfgNode 설정 → dataclass, set_seed 포함,
  옵티마이저 구성 함수를 이 파일로 이동 (로직 동일)
- LSTM 지원: configure_optimizers의 weight decay 분류에 nn.LSTM 분기 추가
  (weight_* → decay, bias_* → no_decay). 
  Transformer/LSTM 노트북이 이 Trainer를 공유할 수 있게 한다.
"""

import time
import random
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader


def set_seed(seed):
    """실험 재현을 위한 랜덤 시드 고정."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TrainerConfig:
    """학습 설정. 기본값은 원본 chargpt 실행 조건과 동일합니다."""
    device: str = "auto"          # 'auto'면 GPU 우선
    num_workers: int = 0          # Colab 호환 (원본 기본값은 4)
    max_iters: int = 2000
    batch_size: int = 64
    learning_rate: float = 5e-4   # 원본 chargpt의 오버라이드 값
    betas: tuple = (0.9, 0.95)
    weight_decay: float = 0.1
    grad_norm_clip: float = 1.0


def configure_optimizers(model, config):
    """AdamW 옵티마이저 생성 (원본 GPT.configure_optimizers와 동일 로직).

    weight decay를 적용할 파라미터(Linear의 weight)와
    적용하지 않을 파라미터(bias, LayerNorm, Embedding)를 분리합니다.
    """
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear,)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = "%s.%s" % (mn, pn) if mn else pn
            if pn.endswith("bias"):
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                no_decay.add(fpn)
            elif isinstance(m, torch.nn.LSTM):
                # LSTM 파라미터: weight_*는 decay, bias_*는 no_decay
                (decay if pn.startswith("weight") else no_decay).add(fpn)

    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, \
        "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert len(param_dict.keys() - union_params) == 0, \
        "parameters %s were not separated into either decay/no_decay set!" \
        % (str(param_dict.keys() - union_params),)

    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))],
         "weight_decay": config.weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        optim_groups, lr=config.learning_rate, betas=config.betas
    )
    return optimizer


class Trainer:
    """단순 학습 루프. GPT에 특화된 내용은 없는 범용 보일러플레이트입니다."""

    def __init__(self, config, model, train_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)

        # 학습 장치 결정
        if config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # 로깅용 상태 변수
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config

        # 옵티마이저 준비
        self.optimizer = configure_optimizers(model, config)

        # 데이터로더 준비 (복원추출 랜덤 샘플링)
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(
                self.train_dataset, replacement=True, num_samples=int(1e10)
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

            # 다음 배치 (x, y) 가져오기
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            # 순전파: 예측과 loss 계산
            _, self.loss = model(x, y)

            # 역전파: 기울기 계산 후 파라미터 갱신
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_norm_clip
            )
            self.optimizer.step()

            self.trigger_callbacks("on_batch_end")
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # 종료 조건
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break

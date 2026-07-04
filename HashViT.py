import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from Network import *
from utils.tools import *


torch.multiprocessing.set_sharing_strategy("file_system")


COMMON_CONFIG = {
    "lambda": 0.0001,
    "optimizer": {
        "type": optim.Adam,
        "optim_params": {"weight_decay": 10 ** -5},
    },
    "info": "[Hash_token]",
    "resize_size": 256,
    "crop_size": 224,
    "net": HashViT_Small,
    "Hash_tokens_num": 1,
    "test_map": 10,
    "device": torch.device("cuda:2"),
    "bit_list": [16, 32, 48, 64],
    "proxy_init": "text",
    "shc_center_root": (
        "/data4/liuxinze/Hash/"
        "Deep-Hashing-with-Semantic-Hash-Centers-for-Image-Retrieval/"
        "save/HashCenters"
    ),
}


DATASET_CONFIGS = {
    "cifar10": {
        "lambda_q": 0,
        "lambda_sd": 1,
        "batch_size": 512,
        "dataset": "cifar10",
        "epoch": 300,
    },
    "imagenet": {
        "lambda_q": 1,
        "lambda_sd": 10,
        "batch_size": 512,
        "dataset": "imagenet",
        "epoch": 200,
    },
    "nuswide_81_m": {
        "lambda_q": 0,
        "lambda_sd": 1,
        "batch_size": 128,
        "multi_loss_mode": "positive_set",
        "multi_center_scale": 24,
        "multi_neg_weight": 1.0,
        "dataset": "nuswide_81_m",
        "epoch": 200,
    },
}


TRAINING_VARIANTS = {
    "cifar10": {
        "head_names": ("head", "hashTokenAdapter"),
        "proxy_lr": 1e-3,
    },
    "imagenet": {
        "head_names": ("hash_writer", "hashTokenAdapter"),
        "proxy_lr": 1e-4,
    },
    "nuswide_81_m": {
        "head_names": ("hash_writer", "hashTokenAdapter"),
        "proxy_lr": 1e-4,
    },
}


MULTI_LABEL_DATASETS = {
    "nuswide_81_m",
    "nuswide_21",
    "nuswide_21_m",
    "coco",
}


def get_config(dataset="cifar10"):
    if dataset not in DATASET_CONFIGS:
        choices = ", ".join(DATASET_CONFIGS)
        raise ValueError(
            f"Unsupported dataset '{dataset}'. Choose from: {choices}."
        )

    config = {**COMMON_CONFIG, **DATASET_CONFIGS[dataset]}
    return config_dataset(config)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)


class NativeHash_Loss(nn.Module):
    def __init__(self, config, bit):
        super(NativeHash_Loss, self).__init__()
        self.device = config["device"]
        self.bit = bit
        self.n_class = config["n_class"]
        self.is_single_label = (
            config["dataset"] not in MULTI_LABEL_DATASETS
        )
        self.lambda_q = config.get("lambda_q", 0.1)
        self.lambda_sd = config.get("lambda_sd", 10)

        init_centers = self._init_proxies(config).to(self.device)
        self.proxies = nn.Parameter(init_centers)
        if self.is_single_label:
            self.margin = 0.1
            self.alpha = 32.0
        else:
            self.margin = 0.05
            self.alpha = 16

    def _init_proxies(self, config):
        proxy_init = config.get("proxy_init", "text").lower()
        if proxy_init == "text":
            init_centers = self._load_text_proxies(config)
        elif proxy_init == "shc":
            init_centers = self._load_shc_proxies(config)
        elif proxy_init == "random":
            init_centers = torch.randn(self.n_class, self.bit)
        else:
            raise ValueError(
                f"Unknown proxy_init '{proxy_init}'. "
                "Choose from: text, shc, random."
            )

        init_centers = self._format_proxy_shape(
            init_centers, proxy_init
        )
        print(
            f"proxy_init: {proxy_init}, "
            f"proxy_shape: {tuple(init_centers.shape)}"
        )
        return init_centers

    def _load_text_proxies(self, config):
        if "nuswide" in config["dataset"]:
            proxy_path = (
                "/data4/liuxinze/Hash/DeepHash-pytorch/nuswide/"
                f"nuswide_text_proxies_{self.bit}bit.pt"
            )
        elif "imagenet" in config["dataset"]:
            proxy_path = (
                "/data4/liuxinze/Hash/DeepHash-pytorch/imagenet/"
                f"nuswide_text_proxies_{self.bit}bit.pt"
            )
        elif "cifar" in config["dataset"]:
            proxy_path = (
                "/data4/liuxinze/Hash/DeepHash-pytorch/cifar/"
                f"nuswide_text_proxies_{self.bit}bit.pt"
            )
        else:
            raise ValueError(
                "No text proxy path configured for dataset "
                f"'{config['dataset']}'."
            )
        print(proxy_path)
        return torch.load(proxy_path, map_location="cpu").float()

    def _load_shc_proxies(self, config):
        dataset_names = self._shc_dataset_candidates(config["dataset"])
        shc_center_root = config.get(
            "shc_center_root",
            (
                "/data4/liuxinze/Hash/"
                "Deep-Hashing-with-Semantic-Hash-Centers-for-Image-Retrieval/"
                "save/HashCenters"
            ),
        )
        proxy_paths = [
            os.path.join(
                shc_center_root,
                f"{dataset}_SHC_HashCenters_bit_{self.bit}.pt",
            )
            for dataset in dataset_names
        ]
        proxy_path = next(
            (path for path in proxy_paths if os.path.exists(path)), None
        )
        if proxy_path is None:
            raise FileNotFoundError(
                "SHC center file not found. Tried: "
                + ", ".join(proxy_paths)
                + ". Generate it first or set config['shc_center_root']."
            )
        print(proxy_path)
        init_centers = torch.load(proxy_path, map_location="cpu")
        if not torch.is_tensor(init_centers):
            init_centers = torch.from_numpy(init_centers)
        return init_centers.float()

    @staticmethod
    def _shc_dataset_candidates(dataset):
        dataset = dataset.lower()
        if "cifar" in dataset:
            candidates = ["cifar10"]
        elif "imagenet" in dataset:
            candidates = ["imagenet"]
        elif "nuswide" in dataset:
            candidates = [
                dataset,
                "nuswide_81_m",
                "nuswide_21_m",
                "nuswide_21",
                "nuswide",
            ]
        else:
            candidates = [dataset]
        return list(dict.fromkeys(candidates))

    def _format_proxy_shape(self, init_centers, proxy_init):
        if init_centers.shape == (self.n_class, self.bit):
            return init_centers
        if init_centers.shape == (self.bit, self.n_class):
            return init_centers.t().contiguous()
        raise ValueError(
            f"{proxy_init} proxies should have shape "
            f"{(self.n_class, self.bit)} or "
            f"{(self.bit, self.n_class)}, "
            f"got {tuple(init_centers.shape)}."
        )

    @staticmethod
    def sample_one_positive(y):
        pos_weight = y.float().clamp_min(0)
        empty_rows = pos_weight.sum(dim=1) <= 0
        if empty_rows.any():
            pos_weight = pos_weight.clone()
            pos_weight[empty_rows, 0] = 1.0
        sampled_idx = torch.multinomial(
            pos_weight, num_samples=1
        ).squeeze(1)
        return torch.zeros_like(pos_weight).scatter_(
            1, sampled_idx.unsqueeze(1), 1.0
        )

    def proxy_exp_loss(
        self, cos_sim, pos_mask, neg_mask, neg_weight=1.0
    ):
        pos_sim = cos_sim - self.margin
        pos_exp = torch.exp(-self.alpha * pos_sim)
        valid_class_mask = (pos_mask.sum(dim=0) > 0).float()
        pos_term = torch.log(
            1.0 + torch.sum(pos_exp * pos_mask, dim=0)
        )
        valid_class_count = valid_class_mask.sum()
        pa_pos_loss = torch.sum(
            pos_term * valid_class_mask
        ) / (valid_class_count + 1e-6)

        neg_sim = cos_sim + self.margin
        neg_exp = torch.exp(self.alpha * neg_sim)
        pa_neg_loss = torch.mean(
            torch.log(
                1.0 + torch.sum(neg_exp * neg_mask, dim=0)
            )
        )
        return pa_pos_loss + neg_weight * pa_neg_loss

    def c_loss(self, h, y, config):
        P = F.normalize(self.proxies, p=2, dim=-1)
        X = F.normalize(h, p=2, dim=-1)
        cos_sim = F.linear(X, P)
        y = y.float()

        if self.is_single_label:
            pos_mask = y
            neg_mask = 1.0 - pos_mask
            return self.proxy_exp_loss(cos_sim, pos_mask, neg_mask)

        multi_loss_mode = config.get(
            "multi_loss_mode", "positive_set"
        )
        if multi_loss_mode == "sample_one":
            pos_mask = self.sample_one_positive(y)
            neg_mask = 1.0 - y
            neg_weight = config.get("multi_neg_weight", 1.0)
            return self.proxy_exp_loss(
                cos_sim, pos_mask, neg_mask, neg_weight
            )
        if multi_loss_mode == "positive_set":
            logits = (
                config.get("multi_center_scale", self.alpha)
                * cos_sim
            )
            pos_logits = logits.masked_fill(y <= 0, -1e9)
            log_pos = torch.logsumexp(pos_logits, dim=1)
            log_all = torch.logsumexp(logits, dim=1)
            return -(log_pos - log_all).mean()
        raise ValueError(
            f"Unknown multi_loss_mode: {multi_loss_mode}"
        )

    @staticmethod
    def sd_loss(h, cls_features):
        if cls_features.dim() == 3:
            cls_features = cls_features.mean(dim=1)
        t_norm = F.normalize(
            cls_features.detach(), p=2, dim=-1
        )
        s_norm = F.normalize(h, p=2, dim=-1)
        S_t = torch.mm(t_norm, t_norm.t())
        S_s = torch.mm(s_norm, s_norm.t())
        return F.mse_loss(S_s, S_t)

    def forward(self, u, cls_features, y, config):
        h = u.tanh()
        center_loss = self.c_loss(h, y, config)
        sd_loss = self.sd_loss(h, cls_features)
        q_loss = (h.abs() - 1).pow(2).mean()
        return (
            center_loss,
            self.lambda_q * q_loss,
            self.lambda_sd * sd_loss,
            torch.tensor(0.0, device=self.device),
        )


def _network_name(network):
    return (
        str(network)
        .replace("<class 'network.", "")
        .replace("<class 'Network.", "")
        .replace("'>", "")
    )


def train_val(config, bit):
    device = config["device"]
    (
        train_loader,
        test_loader,
        dataset_loader,
        num_train,
        num_test,
        num_dataset,
    ) = get_data(config)
    config["num_train"] = num_train
    net = config["net"](
        bit, config["Hash_tokens_num"]
    ).to(device)

    criterion = NativeHash_Loss(config, bit).to(
        config["device"]
    )
    variant = TRAINING_VARIANTS[config["dataset"]]

    backbone_params = []
    head_params = []
    for name, param in net.named_parameters():
        if any(
            head_name in name
            for head_name in variant["head_names"]
        ):
            head_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = [
        {
            "params": backbone_params,
            "lr": 1e-5,
            "weight_decay": 1e-5,
        },
        {
            "params": head_params,
            "lr": 1e-4,
            "weight_decay": 1e-5,
        },
        {
            "params": criterion.parameters(),
            "lr": variant["proxy_lr"],
            "weight_decay": 1e-5,
        },
    ]
    optimizer = config["optimizer"]["type"](param_groups)
    Best_mAP = 0

    for epoch in range(config["epoch"]):
        current_time = time.strftime(
            "%H:%M:%S", time.localtime(time.time())
        )
        print(
            "%s[%2d/%2d][%s] bit:%d, dataset:%s, training...."
            % (
                config["info"],
                epoch + 1,
                config["epoch"],
                current_time,
                bit,
                config["dataset"],
            ),
            end="",
        )

        net.train()
        train_loss = 0
        c_loss = 0
        q_loss = 0
        sd_loss = 0
        i_closs = 0
        total_quant_dist = 0

        for image, label, ind in train_loader:
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            u, cls_ = net(image)
            c, q, sd, i_c = criterion(
                u, cls_, label.float(), config
            )
            loss = c + q + sd + i_c
            train_loss += loss.item()
            c_loss += c.item()
            q_loss += q.item()
            sd_loss += sd.item()
            i_closs += i_c.item()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                batch_dist = (
                    1.0 - u.tanh().abs()
                ).mean().item()
                total_quant_dist += batch_dist

        train_loss = train_loss / len(train_loader)
        c_loss = c_loss / len(train_loader)
        q_loss = q_loss / len(train_loader)
        sd_loss = sd_loss / len(train_loader)
        i_closs = i_closs / len(train_loader)
        avg_quant_dist = total_quant_dist / len(train_loader)
        print(
            "\b\b\b\b\b\b\b loss:%.3f" % train_loss,
            "c:%.3f" % c_loss,
            "q:%.3f" % q_loss,
            "sd:%.3f" % sd_loss,
            f"i_c:{i_closs:.4f}",
            f"Dist:{avg_quant_dist:.4f}",
        )

        if (epoch + 1) % config["test_map"] == 0:
            Best_mAP = validate(
                config,
                Best_mAP,
                test_loader,
                dataset_loader,
                net,
                bit,
                epoch,
                num_dataset,
            )

    return Best_mAP


def parse_args():
    parser = argparse.ArgumentParser(description="Train HashViT.")
    parser.add_argument(
        "--dataset",
        choices=DATASET_CONFIGS,
        default="cifar10",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--bits", type=int, nargs="+", default=None
    )
    parser.add_argument(
        "--proxy-init",
        choices=("text", "shc", "random"),
        default=None,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = get_config(args.dataset)
    if args.device is not None:
        config["device"] = torch.device(args.device)
    if args.bits is not None:
        config["bit_list"] = args.bits
    if args.proxy_init is not None:
        config["proxy_init"] = args.proxy_init

    print(config)
    for bit in config["bit_list"]:
        network_name = _network_name(config["net"])
        config["pr_curve_path"] = (
            f"log/{network_name}/"
            f"Hash_token_{config['dataset']}_{bit}.json"
        )
        Best_mAP = train_val(config, bit)
        os.makedirs(f"log/{config['dataset']}", exist_ok=True)
        path = (
            f"log/{config['dataset']}/{network_name}.txt"
        )
        with open(path, "a", encoding="utf-8") as f:
            f.write(
                f"Bit:{bit}, mAP:{Best_mAP:.5f}, "
                f"Hash_token_num:{config['Hash_tokens_num']}\n"
            )


if __name__ == "__main__":
    main()

import timm
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torchvision import models


__all__ = ["AlexNet", "ResNet", "HashViT_Small", "HashViT_Tiny"]


class AlexNet(nn.Module):
    def __init__(self, hash_bit, pretrained=True):
        super(AlexNet, self).__init__()

        model_alexnet = models.alexnet(pretrained=pretrained)
        self.features = model_alexnet.features
        cl1 = nn.Linear(256 * 6 * 6, 4096)
        cl1.weight = model_alexnet.classifier[1].weight
        cl1.bias = model_alexnet.classifier[1].bias

        cl2 = nn.Linear(4096, 4096)
        cl2.weight = model_alexnet.classifier[4].weight
        cl2.bias = model_alexnet.classifier[4].bias

        self.hash_layer = nn.Sequential(
            nn.Dropout(),
            cl1,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            cl2,
            nn.ReLU(inplace=True),
            nn.Linear(4096, hash_bit),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.hash_layer(x)
        return x, None, None


resnet_dict = {
    "ResNet18": models.resnet18,
    "ResNet34": models.resnet34,
    "ResNet50": models.resnet50,
    "ResNet101": models.resnet101,
    "ResNet152": models.resnet152,
}


class ResNet(nn.Module):
    def __init__(self, hash_bit, res_model="ResNet50"):
        super(ResNet, self).__init__()
        model_resnet = resnet_dict[res_model](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.avgpool,
        )

        self.hash_layer = nn.Linear(model_resnet.fc.in_features, hash_bit)
        self.hash_layer.weight.data.normal_(0, 0.01)
        self.hash_layer.bias.data.fill_(0.0)

    def forward(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        y = self.hash_layer(x)
        return y, None, None


def _load_local_or_pretrained(model_name, pretrained=True, ckpt_path=None):
    model = timm.create_model(
        model_name,
        pretrained=pretrained if ckpt_path is None else False,
        num_classes=0,
    )
    if ckpt_path is not None:
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=False)
    return model


class HashTokenAdapter(nn.Module):
    def __init__(self, dim=384, bit=64, hidden_dim=192, dropout=0.2):
        super().__init__()
        self.bit = bit
        self.ws_dim = dim - bit
        self.ws_to_reg = nn.Sequential(
            nn.LayerNorm(self.ws_dim),
            nn.Linear(self.ws_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bit),
        )

    def forward(self, x):
        reg = x[:, :self.bit]
        workspace = x[:, self.bit:]
        delta_ws = self.ws_to_reg(workspace)
        return (
            torch.cat([reg + delta_ws, workspace], dim=-1),
            reg + delta_ws,
            workspace,
        )


class HashViT_Small(nn.Module):
    def __init__(
        self,
        bit,
        k,
        pretrained=True,
        num_classes=1000,
        img_size=224,
        patch_size=16,
        ckpt_path="/data4/liuxinze/pytorch_model.bin",
        use_checkpoint=True,
    ):
        super(HashViT_Small, self).__init__()
        self.vit = _load_local_or_pretrained(
            "vit_small_patch16_224", pretrained, ckpt_path
        )
        self.patch_embed = self.vit.patch_embed
        self.pos_embed = self.vit.pos_embed
        self.pos_drop = self.vit.pos_drop
        self.blocks = self.vit.blocks
        self.norm = self.vit.norm
        self.bit = bit
        self.k = k
        self.top_k = 3
        self.use_checkpoint = use_checkpoint
        self.cls_token = self.vit.cls_token
        cloned_cls = self.vit.cls_token.clone().detach().repeat(1, self.k, 1)
        self.hash_tokens = nn.Parameter(
            cloned_cls + torch.randn_like(cloned_cls) * 0.01
        )
        self.tanh = nn.Tanh()
        self._extend_position_embedding_smartly()
        self.hashTokenAdapter = HashTokenAdapter(
            dim=self.vit.embed_dim,
            bit=self.bit,
            hidden_dim=192,
            dropout=0.2,
        )

    def _run_block(self, block, x):
        if self.use_checkpoint and self.training and x.requires_grad:
            try:
                return checkpoint(block, x, use_reentrant=False)
            except TypeError:
                return checkpoint(block, x)
        return block(x)

    def _extend_position_embedding_smartly(self):
        original_pe = self.vit.pos_embed
        _, L_original, C = original_pe.shape
        L_new = L_original + self.k
        new_pe = torch.zeros(1, L_new, C, device=original_pe.device)
        new_pe[:, 0, :] = original_pe[:, 0, :]
        patch_mean = original_pe[:, 1:, :].mean(dim=1, keepdim=True)
        for i in range(1, self.k + 1):
            noise = (
                torch.randn(1, 1, C, device=original_pe.device) * 0.02
            )
            new_pe[:, i:i + 1, :] = patch_mean + noise
        new_pe[:, self.k + 1:, :] = original_pe[:, 1:, :]
        self.pos_embed = nn.Parameter(new_pe)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        hash_tokens = [
            self.cls_token.expand(B, -1, -1),
            self.hash_tokens.expand(B, -1, -1),
        ]
        x = torch.cat([torch.cat(hash_tokens, dim=1), x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        writeback_token = x[:, 1, :]
        for _, block in enumerate(self.blocks):
            x = self._run_block(block, x)
            hash_token_after = x[:, 1, :]
            writeback_token, _, _ = self.hashTokenAdapter(
                hash_token_after
            )
            x = torch.cat(
                [x[:, :1, :], writeback_token.unsqueeze(1), x[:, 2:, :]],
                dim=1,
            )
        hash_code = writeback_token[:, :self.bit]
        cls_features = x[:, 0, :].detach()
        return hash_code, cls_features


class HashViT_Tiny(HashViT_Small):
    def __init__(
        self,
        bit,
        k,
        pretrained=True,
        num_classes=1000,
        img_size=224,
        patch_size=16,
        ckpt_path=None,
        use_checkpoint=True,
    ):
        nn.Module.__init__(self)
        self.vit = _load_local_or_pretrained(
            "vit_tiny_patch16_224", pretrained, ckpt_path
        )
        self.patch_embed = self.vit.patch_embed
        self.pos_embed = self.vit.pos_embed
        self.pos_drop = self.vit.pos_drop
        self.blocks = self.vit.blocks
        self.norm = self.vit.norm
        self.bit = bit
        self.k = k
        self.top_k = 3
        self.use_checkpoint = use_checkpoint
        self.cls_token = self.vit.cls_token
        cloned_cls = self.vit.cls_token.clone().detach().repeat(1, self.k, 1)
        self.hash_tokens = nn.Parameter(
            cloned_cls + torch.randn_like(cloned_cls) * 0.01
        )
        self.tanh = nn.Tanh()
        self._extend_position_embedding_smartly()
        self.hashTokenAdapter = HashTokenAdapter(
            dim=self.vit.embed_dim,
            bit=self.bit,
            hidden_dim=192,
            dropout=0.2,
        )

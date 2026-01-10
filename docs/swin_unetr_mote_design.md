# Swin UNETRへのMoTE適用設計

本ドキュメントでは、MoTE（Mixture of Task-specific Experts）フレームワークをSwin UNETRに適用し、臓器ごとの専門家（Adapter）を訓練する方法について説明します。

## 1. 背景

### MoTEの概要
MoTEは、クラス増分学習のためのタスク固有専門家ミクスチャフレームワークです。各タスクごとに独立したAdapter（専門家）を訓練・保存し、テスト時に全専門家で推論して投票・融合を行います。

### 適用目標
- CT画像の臓器セグメンテーションに適用
- 臓器ごとに専門家（Adapter）を訓練
- 新しい臓器を追加しても過去の臓器を忘れない増分学習

## 2. 全体アーキテクチャ

```
入力3Dボリューム [B, 1, D, H, W]
        ↓
┌──────────────────────────────────────┐
│     Swin Transformer Encoder         │
│  ┌─────────────────────────────────┐ │
│  │ Stage 1: [Swin Block + Adapter] │ │
│  │ Stage 2: [Swin Block + Adapter] │ │
│  │ Stage 3: [Swin Block + Adapter] │ │
│  │ Stage 4: [Swin Block + Adapter] │ │
│  └─────────────────────────────────┘ │
└──────────────────────────────────────┘
        ↓
┌──────────────────────────────────────┐
│        UNet Decoder                   │
│  (Skip connections + Upsampling)     │
└──────────────────────────────────────┘
        ↓
┌──────────────────────────────────────┐
│  臓器別専門家の融合（MoTE）           │
│  - 各臓器Adapterで推論               │
│  - ボクセル単位で投票・融合           │
└──────────────────────────────────────┘
        ↓
セグメンテーションマスク [B, num_organs, D, H, W]
```

## 3. Adapterの次元依存性について

### 結論：Adapter自体のコード変更は不要

元のMoTE Adapterは`nn.Linear`を使用しており、次元（2D/3D）に依存しません。

```python
class Adapter(nn.Module):
    def __init__(self, d_model, bottleneck):
        self.down_proj = nn.Linear(d_model, bottleneck)  # 768 → 64
        self.up_proj = nn.Linear(bottleneck, d_model)    # 64 → 768

    def forward(self, x):  # x: [B, N, C]
        down = self.down_proj(x)   # [B, N, 64]
        down = self.non_linear_func(down)
        up = self.up_proj(down)    # [B, N, 768]
        return x + up
```

### Transformerはシーケンス処理

| 入力タイプ | パッチ化後の形状 | Adapterへの入力 |
|-----------|-----------------|----------------|
| 2D画像 `[B,3,224,224]` | `[B, 196, 768]` (14×14パッチ) | `[B, N, C]` |
| 3D CT `[B,1,96,96,96]` | `[B, 216, 768]` (6×6×6パッチ) | `[B, N, C]` |

`nn.Linear`はシーケンス長`N`に依存せず、特徴次元`C`のみに作用するため、2Dでも3Dでも同じコードで動作します。

### 変更が必要な部分と不要な部分

| コンポーネント | 変更要否 | 理由 |
|---------------|---------|------|
| **Adapter** | 不要 | `nn.Linear`は次元に依存しない |
| **パッチ埋め込み** | 必要 | Swin UNETRを使えばOK |
| **位置埋め込み** | 必要 | Swin UNETRを使えばOK |
| **専門家融合** | 変更 | サンプル単位→ボクセル単位 |
| **訓練ループ** | 変更 | セグメンテーション用損失関数 |

## 4. 実装コード

### 4.1 3D医療画像用Adapter

```python
import torch
import torch.nn as nn
import copy
from monai.networks.nets import SwinUNETR

class Adapter3D(nn.Module):
    """3D医療画像用Adapter（MoTEから継承）"""
    def __init__(self, d_model=768, bottleneck=64, dropout=0.1):
        super().__init__()
        self.down_proj = nn.Linear(d_model, bottleneck)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(bottleneck, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        # LoRA風初期化
        nn.init.kaiming_uniform_(self.down_proj.weight, a=5**0.5)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True):
        residual = x
        x = self.layer_norm(x)
        x = self.down_proj(x)
        x = self.non_linear_func(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        if add_residual:
            x = x + residual
        return x
```

### 4.2 臓器別専門家を持つSwin UNETR

```python
class SwinUNETRWithMoTE(nn.Module):
    """臓器別専門家を持つSwin UNETR"""

    def __init__(
        self,
        img_size=(96, 96, 96),
        in_channels=1,
        feature_size=48,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        bottleneck_size=64,
        organs=None,
    ):
        super().__init__()

        self.organs = organs or ['liver', 'spleen', 'kidney', 'pancreas']
        self.num_organs = len(self.organs)

        # ベースSwin UNETR（事前学習済み）
        self.swin_unetr = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=self.num_organs + 1,  # +1 for background
            feature_size=feature_size,
            depths=depths,
            num_heads=num_heads,
            use_checkpoint=True,
        )

        # Swinエンコーダの次元数（各ステージ）
        self.embed_dims = [
            feature_size,           # Stage 1: 48
            feature_size * 2,       # Stage 2: 96
            feature_size * 4,       # Stage 3: 192
            feature_size * 8,       # Stage 4: 384
        ]

        # 現在訓練中のAdapter（1臓器分）
        self.cur_adapters = self._create_adapter_set(bottleneck_size)

        # 保存済みAdapter（臓器ごとのリスト）
        self.adapter_list = []
        self.organ_names = []

        # ベースモデルをフリーズ
        self._freeze_base_model()

    def _create_adapter_set(self, bottleneck_size):
        """各Swinステージ用のAdapterセットを作成"""
        adapters = nn.ModuleDict()
        for stage_idx, dim in enumerate(self.embed_dims):
            adapters[f'stage_{stage_idx}'] = Adapter3D(
                d_model=dim,
                bottleneck=bottleneck_size
            )
        return adapters

    def _freeze_base_model(self):
        """ベースSwin UNETRをフリーズ"""
        for param in self.swin_unetr.parameters():
            param.requires_grad = False
        for param in self.cur_adapters.parameters():
            param.requires_grad = True

    def add_organ_expert(self, organ_name):
        """現在のAdapterを保存し、新しいAdapterを作成"""
        self.adapter_list.append(
            copy.deepcopy(self.cur_adapters.requires_grad_(False))
        )
        self.organ_names.append(organ_name)
        self.cur_adapters = self._create_adapter_set(64)
        self._freeze_base_model()

    def forward_with_adapter(self, x, adapters):
        """特定のAdapterセットで推論"""
        swin_encoder = self.swin_unetr.swinViT

        hidden_states = swin_encoder.patch_embed(x)
        hidden_states = swin_encoder.pos_drop(hidden_states)

        encoder_outputs = []
        for stage_idx, layer in enumerate(swin_encoder.layers):
            hidden_states = layer(hidden_states)
            adapter = adapters[f'stage_{stage_idx}']
            hidden_states = adapter(hidden_states)
            encoder_outputs.append(hidden_states)

        out = self.swin_unetr.decoder(encoder_outputs)
        return out

    def forward_train(self, x):
        """訓練時：現在のAdapterのみ使用"""
        return self.forward_with_adapter(x, self.cur_adapters)

    def forward_test(self, x):
        """テスト時：全専門家で推論→融合"""
        all_outputs = []
        all_confidences = []

        for adapters in self.adapter_list:
            out = self.forward_with_adapter(x, adapters)
            all_outputs.append(out)
            conf = torch.softmax(out, dim=1).max(dim=1)[0]
            all_confidences.append(conf)

        cur_out = self.forward_with_adapter(x, self.cur_adapters)
        all_outputs.append(cur_out)
        cur_conf = torch.softmax(cur_out, dim=1).max(dim=1)[0]
        all_confidences.append(cur_conf)

        final_output = self.merge_expert_outputs(all_outputs, all_confidences)
        return final_output

    def merge_expert_outputs(self, outputs, confidences):
        """ボクセル単位で専門家出力を融合"""
        stacked_outputs = torch.stack(outputs, dim=0)
        stacked_confs = torch.stack(confidences, dim=0)

        num_experts = len(outputs)
        max_conf_idx = stacked_confs.argmax(dim=0)

        weights = stacked_confs.clone()
        for exp_idx in range(num_experts):
            mask = (max_conf_idx != exp_idx)
            weights[exp_idx][mask] *= 0.1

        weights = torch.softmax(weights, dim=0)
        weights = weights.unsqueeze(2)
        merged = (stacked_outputs * weights).sum(dim=0)

        return merged

    def forward(self, x, test=False):
        if test:
            return self.forward_test(x)
        else:
            return self.forward_train(x)
```

### 4.3 臓器増分学習のトレーニングループ

```python
from monai.losses import DiceCELoss

class OrganIncrementalTrainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def train_organ(self, organ_name, train_loader, epochs=100):
        """1つの臓器に対して専門家を訓練"""
        print(f"Training expert for: {organ_name}")

        optimizer = torch.optim.AdamW(
            self.model.cur_adapters.parameters(),
            lr=1e-4
        )
        criterion = DiceCELoss(to_onehot_y=True, softmax=True)

        for epoch in range(epochs):
            for batch in train_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(images, test=False)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.model.add_organ_expert(organ_name)

    def train_all_organs(self, organ_datasets):
        """全臓器を順次訓練"""
        for organ_name, train_loader in organ_datasets.items():
            self.train_organ(organ_name, train_loader)
            print(f"Completed: {organ_name}")
            print(f"Total experts: {len(self.model.adapter_list)}")
```

## 5. 使用例

```python
organs = ['liver', 'spleen', 'kidney_left', 'kidney_right', 'pancreas']

model = SwinUNETRWithMoTE(
    img_size=(96, 96, 96),
    in_channels=1,
    feature_size=48,
    organs=organs,
)

# 事前学習済み重みをロード
model.swin_unetr.load_state_dict(
    torch.load('swin_unetr_pretrained.pth'),
    strict=False
)

trainer = OrganIncrementalTrainer(model, device='cuda')

# 臓器ごとに順次訓練
organ_datasets = {
    'liver': liver_loader,
    'spleen': spleen_loader,
    'kidney': kidney_loader,
    'pancreas': pancreas_loader,
}
trainer.train_all_organs(organ_datasets)

# テスト時（全専門家で推論）
model.eval()
output = model(test_image, test=True)
```

## 6. 設計のポイント

| 要素 | 元MoTE（分類） | Swin UNETR版（セグメンテーション） |
|------|---------------|----------------------------------|
| Adapter配置 | 各Transformerブロック | 各Swinステージ（4段階） |
| 次元数 | 768固定 | 48→96→192→384（段階的） |
| 融合単位 | サンプル単位 | ボクセル単位 |
| タスク定義 | クラス範囲 | 臓器ラベル |
| 出力 | ロジット（分類） | セグメンテーションマスク |

## 7. 医療画像特有の考慮事項

1. **ボクセル単位融合**: セグメンテーションでは各ボクセルで最適な専門家を選択
2. **臓器間の重複**: 隣接臓器の境界では複数専門家の意見を考慮
3. **メモリ効率**: 3Dボリュームは大きいため、チェックポイント技術を使用
4. **事前学習活用**: BTCVやTotalSegmentator等で事前学習したSwin UNETRをベースに

## 8. 主な利点

1. **壊滅的忘却回避**: 各臓器のAdapterを独立保存
2. **パラメータ効率**: LoRA風の瓶首設計でパラメータ削減
3. **スケーラビリティ**: 臓器数に線形でAdapter増加
4. **柔軟な融合**: 複数の投票・融合戦略から選択可能
5. **事前学習の活用**: Swin UNETR事前学習の知識を効果的に転用

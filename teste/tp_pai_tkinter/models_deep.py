# models_deep.py
import torch
import torch.nn as nn
from torchvision import models


def load_resnet50():
    """
    Cria uma ResNet50 base.
    Não usamos pesos pré-treinados porque o state_dict treinado
    vai sobrescrever tudo de qualquer forma.
    """
    model = models.resnet50(weights=None)
    return model


def make_extra_layer(in_features: int, extra_hidden: int, n_layers: int, use_extra: bool):
    """
    Cria a camada 'extra' usada no DeepClassifier.

    Retorna:
      - um nn.Module (Sequential ou Identity)
      - o número de features de saída (out_features)
    """
    if not use_extra or n_layers <= 0:
        # Sem camada extra: passa direto
        return nn.Identity(), in_features

    layers = []
    input_dim = in_features

    for i in range(n_layers):
        layers.append(nn.Linear(input_dim, extra_hidden))
        layers.append(nn.ReLU(inplace=True))
        input_dim = extra_hidden

    return nn.Sequential(*layers), input_dim


class DeepClassifier(nn.Module):
    """Classe base para instanciar classificador profundo"""

    def __init__(
        self,
        num_classes: int = 2,
        extra_hidden: int = 256,
        n_layers: int = 1,
        use_extra: bool = True,
        class_weights=None
    ):
        super().__init__()

        base = load_resnet50()
        # backbone = tudo menos a FC final
        self.backbone = nn.Sequential(*(list(base.children())[:-1]))

        self.extra, out_features = make_extra_layer(
            in_features=2048,
            extra_hidden=extra_hidden,
            n_layers=n_layers,
            use_extra=use_extra
        )

        self.fc = nn.Linear(out_features, num_classes)

        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, x):
        x = self.backbone(x)          # [B, 2048, 1, 1]
        x = torch.flatten(x, 1)       # [B, 2048]
        x = self.extra(x)             # [B, out_features]
        return self.fc(x)             # [B, num_classes]

    def step(self, batch):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        return loss, preds

class DeepRegressor(nn.Module):
    """Classe base para instanciar regressor profundo"""

    def __init__(self, extra_hidden=256, n_layers=1, use_extra=True):
        super().__init__()

        base = load_resnet50()
        self.backbone = nn.Sequential(*(list(base.children())[:-1]))

        self.extra, out_features = make_extra_layer(
            in_features=2048,
            extra_hidden=extra_hidden,
            n_layers=n_layers,
            use_extra=use_extra
        )
        self.fc = nn.Linear(out_features, 1)

        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.extra(x)
        # [B, 1] -> [B]
        return self.fc(x).squeeze(1)

    def step(self, batch):
        x, y = batch
        y = y.float()
        preds = self(x)
        loss = self.loss_fn(preds, y)
        return loss, preds

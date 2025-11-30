import torch
from models_deep import DeepClassifier

# 1) Recria o modelo com os MESMOS hiperparâmetros do treino
# (agora com use_extra=True para bater com os pesos extra.*)
deep_classifier = DeepClassifier(
    num_classes=2,
    extra_hidden=64,
    n_layers=5,
    use_extra=True,   # <<< IMPORTANTE: TRUE
    class_weights=None
).to("cpu")

# 2) Carrega o state_dict salvo no treino
state_dict = torch.load("classifier_nn.pth", map_location="cpu")

# 3) Carrega os pesos, ignorando chaves que não existem no modelo atual
# (por ex. loss_fn.weight, que não é usado na inferência)
missing, unexpected = deep_classifier.load_state_dict(state_dict, strict=False)
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)

# 4) Coloca em modo avaliação
deep_classifier.eval()

# 5) Salva o MODELO COMPLETO pronto pra interface
torch.save(deep_classifier, "classifier_nn_full.pth")

print("Modelo completo salvo em classifier_nn_full.pth")

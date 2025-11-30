# converter_regressor.py
import torch
from models_deep import DeepRegressor

# 1) Recria o modelo com os MESMOS hiperparâmetros do treino
# (se no treino você mudou extra_hidden/n_layers/use_extra, ajuste aqui!)
deep_regressor = DeepRegressor(
    extra_hidden=256,
    n_layers=1,
    use_extra=True
).to("cpu")

# 2) Carrega o state_dict salvo no treino
state_dict = torch.load("regressor_nn.pth", map_location="cpu")

# 3) Carrega os pesos; strict=False ignora chaves extras que não importam
missing, unexpected = deep_regressor.load_state_dict(state_dict, strict=False)
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)

# 4) Coloca em modo avaliação
deep_regressor.eval()

# 5) Salva o MODELO COMPLETO pronto pra interface
torch.save(deep_regressor, "regressor_nn_full.pth")

print("Modelo completo salvo em regressor_nn_full.pth")

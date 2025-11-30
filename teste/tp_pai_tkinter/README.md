# Segmentação de Ventriculos Cerebrais

Interface gráfica em Tkinter para segmentação de ventrículos em exames de imagem (PNG/JPG/NIfTI). O aplicativo permite carregar exames, aplicar diferentes técnicas de segmentação e visualizar métricas geométricas calculadas a partir dos contornos resultantes.

## Requisitos
- Python 3.10+ com Tkinter disponível (já incluso na maior parte das distribuições).
- Dependências listadas em [`requirements.txt`](requirements.txt). Instale com:
  ```bash
  pip install -r requirements.txt
  ```

## Como executar
1. Certifique-se de que o ambiente virtual (opcional) esteja ativo e as dependências instaladas.
2. Inicie a aplicação GUI com:
   ```bash
   python interface_tkinter.py
   ```
3. Use o menu **Arquivo → Abrir** para selecionar uma imagem PNG/JPG ou NIfTI (.nii/.nii.gz). Se optar por NIfTI, é necessário ter o `nibabel` instalado.
4. Escolha o método de segmentação desejado na barra lateral (Canny, Watershed, Otsu, K-means ou crescimento de regiões) e visualize o resultado.
5. As métricas calculadas (área total, circularidade média, excentricidade, perímetro, solidez e razão de aspecto) aparecem no painel de descritores.

## Observações
- Alguns modelos treinados são carregados por `treatment_flow.py`; verifique se os arquivos em `modelos/` e `extracted_images/` estão presentes quando necessário.
- Para datasets de exemplo, consulte os arquivos `oasis_longitudinal_demographic.csv`, `oasis_with_ventricle_features.csv` ou `p1.csv` incluídos no repositório.

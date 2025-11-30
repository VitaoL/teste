"""
Módulo para visualização do fluxo de processamento de segmentação de ventrículos.

Este módulo contém funções para documentar visualmente o pipeline completo
de processamento, desde a imagem original até a detecção final dos ventrículos.
"""

import os
import cv2
import numpy as np


def load_first_image(input_directory):
    """Carrega a primeira imagem válida do diretório."""
    files = os.listdir(input_directory)

    for file_name in files:
        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue

        file_path = os.path.join(input_directory, file_name)
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        if image is not None:
            return image, file_name

    raise FileNotFoundError(f"Nenhuma imagem válida encontrada em {input_directory}")


def create_brain_mask(image, threshold=20):
    """Cria máscara binária do cérebro (remove fundo preto)."""
    _, brain_mask = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return brain_mask


def detect_dark_areas(image, threshold=82):
    """Detecta áreas escuras na imagem (threshold invertido)."""
    _, dark_areas = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    return dark_areas


def find_contours(binary_image):
    """Encontra contornos em imagem binária."""
    contours, hierarchy = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours, hierarchy


def get_largest_contour(contours):
    """
    Retorna os maiores contornos que formam os ventrículos laterais.
    Filtra contornos muito pequenos ou muito distantes entre si.
    """
    if len(contours) == 0:
        return None

    # Ordenar por área
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Se só tem 1 contorno, retornar ele
    if len(sorted_contours) == 1:
        # Verificar se não é muito pequeno (ruído)
        if cv2.contourArea(sorted_contours[0]) >= 3000:
            return [sorted_contours[0]]
        return None

    # Pegar o maior contorno (sempre válido)
    main_contour = sorted_contours[0]
    main_area = cv2.contourArea(main_contour)

    # Se o maior for muito pequeno, retornar None
    if main_area < 3000:
        return None

    result = [main_contour]

    # Calcular centroide do maior contorno
    M_main = cv2.moments(main_contour)
    if M_main['m00'] == 0:
        return result

    main_cx = M_main['m10'] / M_main['m00']
    main_cy = M_main['m01'] / M_main['m00']

    # Verificar o segundo maior
    if len(sorted_contours) > 1:
        second_contour = sorted_contours[1]
        second_area = cv2.contourArea(second_contour)

        # Só incluir o segundo se:
        # 1. Tiver área razoável (>= 30% do maior OU >= 3000px²)
        # 2. Estiver próximo do primeiro (distância < 150px entre centroides)

        if second_area >= max(main_area * 0.3, 3000):
            M_second = cv2.moments(second_contour)
            if M_second['m00'] > 0:
                second_cx = M_second['m10'] / M_second['m00']
                second_cy = M_second['m01'] / M_second['m00']

                # Calcular distância entre centroides
                distance = np.sqrt((second_cx - main_cx)**2 + (second_cy - main_cy)**2)

                # Ventrículos laterais ficam lado a lado (distância horizontal razoável)
                # Mas não muito longe verticalmente
                horizontal_dist = abs(second_cx - main_cx)
                vertical_dist = abs(second_cy - main_cy)

                # Aceitar se:
                # - Distância total < 150px (estruturas próximas)
                # - Ou distância horizontal > vertical (lado a lado) E distância total < 200px
                if distance < 150 or (horizontal_dist > vertical_dist and distance < 200):
                    result.append(second_contour)

    return result

def circular_crop(image, radius=250):
    """Cria máscara circular na região central."""
    h, w = image.shape
    cx, cy = w // 2, h // 2

    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    mask = dist <= radius

    cropped = np.zeros_like(image)
    cropped[mask] = image[mask]

    return cropped


def segment_ventricles_canny(image, crop_radius=250, low_threshold=20, high_threshold=60):
    """
    Segmenta ventrículos usando detecção de bordas Canny com heurística de área escura.

    Args:
        image: Imagem em escala de cinza
        crop_radius: Raio do crop circular
        low_threshold: Threshold baixo para Canny
        high_threshold: Threshold alto para Canny

    Returns:
        Tuple com (imagem cropada, bordas detectadas, contornos, ventrículos conectados)
    """
    # 1. Crop circular
    cropped = circular_crop(image, radius=crop_radius)

    # 2. HEURÍSTICA: Primeiro identificar regiões escuras (ventrículos são escuros)
    _, dark_areas = cv2.threshold(cropped, 85, 255, cv2.THRESH_BINARY_INV)
    _, valid_area = cv2.threshold(cropped, 10, 255, cv2.THRESH_BINARY)
    dark_areas = cv2.bitwise_and(dark_areas, valid_area)

    # 3. Aplicar morfologia para limpar ruído nas áreas escuras
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dark_areas_cleaned = cv2.morphologyEx(dark_areas, cv2.MORPH_OPEN, kernel_clean)

    # 4. Fechar gaps nas áreas escuras
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    dark_areas_closed = cv2.morphologyEx(dark_areas_cleaned, cv2.MORPH_CLOSE, kernel_close)

    # 5. Aplicar blur apenas nas regiões escuras para Canny
    cropped_masked = cv2.bitwise_and(cropped, cropped, mask=dark_areas_closed)
    blurred = cv2.GaussianBlur(cropped_masked, (5, 5), 0)

    # 6. Detecção de bordas Canny (apenas nas regiões escuras)
    edges = cv2.Canny(blurred, low_threshold, high_threshold)

    # 7. Aplicar máscara de áreas escuras nas bordas
    edges_in_dark = cv2.bitwise_and(edges, dark_areas_closed)

    # 8. Dilatar bordas para conectar fragmentos
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges_dilated = cv2.dilate(edges_in_dark, kernel_dilate, iterations=3)

    # 9. Fechar contornos para criar regiões sólidas
    kernel_close_edges = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    edges_closed = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel_close_edges)

    # 10. Encontrar contornos nas bordas fechadas
    contours_edges, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 11. Preencher os contornos encontrados
    filled_mask = np.zeros_like(edges_closed)
    if len(contours_edges) > 0:
        cv2.drawContours(filled_mask, contours_edges, -1, 255, -1)

    # 12. HEURÍSTICA: Combinar com áreas escuras originais
    # Isso garante que pegamos áreas escuras mesmo se Canny não detectou bordas perfeitas
    combined = cv2.bitwise_or(filled_mask, dark_areas_closed)

    # 13. Aplicar fechamento morfológico final para conectar ventrículos
    k_h = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 5))
    ventricles_connected = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_h)
    k_v = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 10))
    ventricles_connected = cv2.morphologyEx(ventricles_connected, cv2.MORPH_CLOSE, k_v)

    # 14. Encontrar contornos finais
    contours, _ = cv2.findContours(ventricles_connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 15. HEURÍSTICA: Pegar apenas a MAIOR ÁREA ESCURA CENTRAL
    h, w = cropped.shape
    cx, cy = w / 2, h / 2

    # Primeiro filtrar por proximidade ao centro
    central_contours = []
    max_allowed_dist = crop_radius * 0.5  # Apenas regiões bem centrais

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Filtrar por área mínima (evita ruídos pequenos)
        if area < 5000:
            continue

        # Calcular centroide
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue

        cnt_cx = M['m10'] / M['m00']
        cnt_cy = M['m01'] / M['m00']

        # Distância ao centro
        dist = np.sqrt((cnt_cx - cx)**2 + (cnt_cy - cy)**2)

        # HEURÍSTICA: Aceitar apenas contornos MUITO próximos ao centro
        if dist <= max_allowed_dist:
            central_contours.append({
                'contour': cnt,
                'area': area,
                'distance': dist,
                'centroid': (cnt_cx, cnt_cy)
            })

    # Ordenar por área (maior primeiro)
    central_contours.sort(key=lambda x: x['area'], reverse=True)

    # Pegar apenas a MAIOR região central (ventrículos laterais formam uma massa)
    # Ou os 2 maiores se forem de tamanhos similares (ventrículos esquerdo e direito)
    filtered_contours = []
    if len(central_contours) > 0:
        # Sempre pegar o maior
        largest = central_contours[0]
        filtered_contours.append(largest['contour'])

        # Se houver um segundo contorno com área similar (>50% do maior), incluir
        if len(central_contours) > 1:
            second = central_contours[1]
            area_ratio = second['area'] / largest['area']

            # Se o segundo é pelo menos 50% do tamanho do maior, provavelmente é o outro ventrículo
            if area_ratio > 0.5:
                filtered_contours.append(second['contour'])

    return cropped, edges, filtered_contours, ventricles_connected


def segment_ventricles_watershed(image, crop_radius=250):
    """
    Segmenta ventrículos usando Watershed com marcadores.
    """
    cropped = circular_crop(image, radius=crop_radius)

    # Usar threshold similar ao método threshold original
    _, dark_areas = cv2.threshold(cropped, 85, 255, cv2.THRESH_BINARY_INV)
    _, valid_area = cv2.threshold(cropped, 10, 255, cv2.THRESH_BINARY)
    thresh = cv2.bitwise_and(dark_areas, valid_area)

    # Remover ruído
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Área de fundo certa
    sure_bg = cv2.dilate(opening, kernel, iterations=2)

    # Área de foreground certa (mais permissivo)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Região desconhecida
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marcar regiões
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Aplicar watershed
    cropped_color = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(cropped_color, markers)

    # Criar máscara dos ventrículos (pegar todas as regiões, não só >1)
    ventricles_mask = np.zeros_like(cropped)
    ventricles_mask[markers > 1] = 255

    # Se não achou nada, usar a máscara de opening diretamente
    if ventricles_mask.sum() == 0:
        ventricles_mask = opening.copy()

    # Conectar ventrículos
    k_h = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 5))
    ventricles_mask = cv2.morphologyEx(ventricles_mask, cv2.MORPH_CLOSE, k_h)
    k_v = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 10))
    ventricles_mask = cv2.morphologyEx(ventricles_mask, cv2.MORPH_CLOSE, k_v)

    # Encontrar contornos e filtrar
    contours, _ = cv2.findContours(ventricles_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Aplicar heurística central (mais permissiva)
    h, w = cropped.shape
    cx, cy = w / 2, h / 2
    filtered = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 3000:  # Reduzido de 5000 para 3000
            continue
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cnt_cx = M['m10'] / M['m00']
        cnt_cy = M['m01'] / M['m00']
        dist = np.sqrt((cnt_cx - cx)**2 + (cnt_cy - cy)**2)
        if dist <= crop_radius * 0.7:  # Aumentado de 0.5 para 0.7
            filtered.append({'contour': cnt, 'area': area})

    filtered.sort(key=lambda x: x['area'], reverse=True)
    result = [f['contour'] for f in filtered[:2]]

    return cropped, ventricles_mask, result, ventricles_mask


def segment_ventricles_otsu(image, crop_radius=250):
    """
    Segmenta ventrículos usando Otsu threshold automático.
    """
    cropped = circular_crop(image, radius=crop_radius)

    # Aplicar Otsu para threshold automático
    _, thresh = cv2.threshold(cropped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Máscara de área válida
    _, valid_area = cv2.threshold(cropped, 10, 255, cv2.THRESH_BINARY)
    thresh = cv2.bitwise_and(thresh, valid_area)

    # Morfologia para limpar
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Fechar gaps
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)

    # Conectar ventrículos
    k_h = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 5))
    connected = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, k_h)
    k_v = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 10))
    connected = cv2.morphologyEx(connected, cv2.MORPH_CLOSE, k_v)

    # Encontrar contornos
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrar por centralidade
    h, w = cropped.shape
    cx, cy = w / 2, h / 2
    filtered = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5000:
            continue
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cnt_cx = M['m10'] / M['m00']
        cnt_cy = M['m01'] / M['m00']
        dist = np.sqrt((cnt_cx - cx)**2 + (cnt_cy - cy)**2)
        if dist <= crop_radius * 0.5:
            filtered.append({'contour': cnt, 'area': area})

    filtered.sort(key=lambda x: x['area'], reverse=True)
    result = [f['contour'] for f in filtered[:2]]

    return cropped, thresh, result, connected


def segment_ventricles_kmeans(image, crop_radius=250, k=4):
    """
    Segmenta ventrículos usando K-Means clustering.
    """
    cropped = circular_crop(image, radius=crop_radius)

    # Máscara de área válida (pixels > 10)
    _, valid_area = cv2.threshold(cropped, 10, 255, cv2.THRESH_BINARY)

    # Preparar dados para K-Means (apenas pixels válidos)
    valid_pixels = cropped[valid_area == 255]
    pixel_values = valid_pixels.reshape((-1, 1))
    pixel_values = np.float32(pixel_values)

    # Aplicar K-Means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels_valid, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Identificar cluster mais escuro (ventrículos)
    centers = centers.flatten()
    darkest_cluster = np.argmin(centers)

    # Criar máscara completa
    labels_full = np.full(cropped.shape, -1, dtype=np.int32)
    labels_full[valid_area == 255] = labels_valid.flatten()

    ventricles_mask = np.zeros_like(cropped)
    ventricles_mask[labels_full == darkest_cluster] = 255

    # Morfologia para limpar e conectar
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned = cv2.morphologyEx(ventricles_mask, cv2.MORPH_OPEN, kernel)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)

    # Conectar ventrículos
    k_h = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 7))
    connected = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, k_h)
    k_v = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 15))
    connected = cv2.morphologyEx(connected, cv2.MORPH_CLOSE, k_v)

    # Contornos
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrar (mais permissivo)
    h, w = cropped.shape
    cx, cy = w / 2, h / 2
    filtered = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 3000:  # Reduzido de 5000 para 3000
            continue
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cnt_cx = M['m10'] / M['m00']
        cnt_cy = M['m01'] / M['m00']
        dist = np.sqrt((cnt_cx - cx)**2 + (cnt_cy - cy)**2)
        if dist <= crop_radius * 0.7:  # Aumentado de 0.5 para 0.7
            filtered.append({'contour': cnt, 'area': area})

    filtered.sort(key=lambda x: x['area'], reverse=True)
    result = [f['contour'] for f in filtered[:2]]

    return cropped, ventricles_mask, result, connected


def segment_ventricles_region_growing(image, crop_radius=250):
    """
    Segmenta ventrículos usando Region Growing a partir de múltiplos seeds na região central escura.
    """
    from collections import deque

    cropped = circular_crop(image, radius=crop_radius)
    h, w = cropped.shape

    # Identificar região escura central primeiro
    _, dark_areas = cv2.threshold(cropped, 85, 255, cv2.THRESH_BINARY_INV)
    _, valid_area = cv2.threshold(cropped, 10, 255, cv2.THRESH_BINARY)
    dark_areas = cv2.bitwise_and(dark_areas, valid_area)

    # Limpar ruído
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dark_areas = cv2.morphologyEx(dark_areas, cv2.MORPH_OPEN, kernel_small)

    # Encontrar região central escura para seed
    cy, cx = h // 2, w // 2

    # Procurar por pixel escuro próximo ao centro
    search_radius = 50
    seed_points = []
    for dy in range(-search_radius, search_radius, 10):
        for dx in range(-search_radius, search_radius, 10):
            y, x = cy + dy, cx + dx
            if 0 <= y < h and 0 <= x < w and dark_areas[y, x] > 0:
                seed_points.append((x, y))

    if not seed_points:
        # Fallback: usar centro
        seed_points = [(cx, cy)]

    # Threshold range mais permissivo
    threshold_range = 40

    # Inicializar máscara
    mask = np.zeros_like(cropped, dtype=np.uint8)
    visited = np.zeros_like(cropped, dtype=bool)

    # Region growing de múltiplos seeds
    for seed_x, seed_y in seed_points:
        seed_intensity = int(cropped[seed_y, seed_x])

        queue = deque([(seed_x, seed_y)])
        if not visited[seed_y, seed_x]:
            visited[seed_y, seed_x] = True

            while queue:
                x, y = queue.popleft()

                # Verificar se pixel está na faixa de intensidade e é escuro
                pixel_val = int(cropped[y, x])
                if abs(pixel_val - seed_intensity) <= threshold_range and pixel_val < 100:
                    mask[y, x] = 255

                    # Adicionar vizinhos
                    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                            visited[ny, nx] = True
                            queue.append((nx, ny))

    # Morfologia agressiva para conectar
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)

    # Conectar ventrículos
    k_h = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 8))
    connected = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_h)
    k_v = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 20))
    connected = cv2.morphologyEx(connected, cv2.MORPH_CLOSE, k_v)

    # Contornos
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrar (mais permissivo)
    filtered = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 3000:  # Reduzido de 5000
            continue
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cnt_cx = M['m10'] / M['m00']
        cnt_cy = M['m01'] / M['m00']
        dist = np.sqrt((cnt_cx - cx)**2 + (cnt_cy - cy)**2)
        if dist <= crop_radius * 0.7:  # Aumentado de 0.5
            filtered.append({'contour': cnt, 'area': area})

    filtered.sort(key=lambda x: x['area'], reverse=True)
    result = [f['contour'] for f in filtered[:2]]

    return cropped, mask, result, connected


def compare_all_methods(image_path, crop_radius=250, save_path='comparacao_todos_metodos.png'):
    """
    Compara 6 métodos de segmentação: Threshold, Canny, Watershed, Otsu, K-Means e Region Growing.
    """
    # Carregar imagem
    if isinstance(image_path, str):
        if os.path.isdir(image_path):
            image, filename = load_first_image(image_path)
        else:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            filename = os.path.basename(image_path)
    else:
        image = image_path
        filename = "image"

    print(f"\nProcessando: {filename}")
    print("="*70)

    # Executar todos os métodos
    methods = []

    # 1. Threshold
    cropped = circular_crop(image, radius=crop_radius)
    _, dark_areas = cv2.threshold(cropped, 85, 255, cv2.THRESH_BINARY_INV)
    _, valid_area = cv2.threshold(cropped, 10, 255, cv2.THRESH_BINARY)
    ventricles_threshold = cv2.bitwise_and(dark_areas, valid_area)
    k_h = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 5))
    ventricles_threshold = cv2.morphologyEx(ventricles_threshold, cv2.MORPH_CLOSE, k_h)
    k_v = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 10))
    ventricles_threshold = cv2.morphologyEx(ventricles_threshold, cv2.MORPH_CLOSE, k_v)
    contours_threshold, _ = cv2.findContours(ventricles_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_threshold = get_largest_contour(contours_threshold)
    methods.append(('Threshold', image, ventricles_threshold, largest_threshold, (0, 255, 0)))

    # 2. Canny
    _, _, contours_canny, mask_canny = segment_ventricles_canny(image, crop_radius)
    largest_canny = get_largest_contour(contours_canny)
    methods.append(('Canny', image, mask_canny, largest_canny, (255, 0, 255)))

    # 3. Watershed
    _, mask_watershed, contours_watershed, _ = segment_ventricles_watershed(image, crop_radius)
    largest_watershed = get_largest_contour(contours_watershed)
    methods.append(('Watershed', image, mask_watershed, largest_watershed, (0, 255, 255)))

    # 4. Otsu
    _, mask_otsu, contours_otsu, _ = segment_ventricles_otsu(image, crop_radius)
    largest_otsu = get_largest_contour(contours_otsu)
    methods.append(('Otsu', image, mask_otsu, largest_otsu, (255, 255, 0)))

    # 5. K-Means
    _, mask_kmeans, contours_kmeans, _ = segment_ventricles_kmeans(image, crop_radius)
    largest_kmeans = get_largest_contour(contours_kmeans)
    methods.append(('K-Means', image, mask_kmeans, largest_kmeans, (255, 128, 0)))

    # 6. Region Growing
    _, mask_rg, contours_rg, _ = segment_ventricles_region_growing(image, crop_radius)
    largest_rg = get_largest_contour(contours_rg)
    methods.append(('Region Growing', image, mask_rg, largest_rg, (128, 0, 255)))

    # Criar visualização
    rows = []

    for name, img, mask, contours, color in methods:
        # Resultado com contornos
        result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if contours:
            cv2.drawContours(result, contours, -1, color, 2)

        # Adicionar texto
        cv2.putText(result, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Calcular área
        total_area = sum(cv2.contourArea(c) for c in contours) if contours else 0
        cv2.putText(result, f"Area: {total_area:.0f}px", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        rows.append(result)

        # Estatísticas
        print(f"{name:15} - Contornos: {len(contours) if isinstance(contours, list) else 0:2}, Área: {total_area:8.0f} px²")

    # Criar grid 3x2
    row1 = np.hstack(rows[0:3])
    row2 = np.hstack(rows[3:6])
    final_image = np.vstack([row1, row2])

    # Salvar
    cv2.imwrite(save_path, final_image)
    print(f"\nComparação salva em: {save_path}")
    print("="*70)

    return final_image


def compare_segmentation_methods(image_path, crop_radius=250, save_path='comparacao_metodos.png'):
    """
    Compara os dois métodos de segmentação: Threshold e Canny.

    Args:
        image_path: Caminho da imagem
        crop_radius: Raio do crop circular
        save_path: Onde salvar a comparação

    Returns:
        Imagem comparativa
    """
    # Carregar imagem
    if isinstance(image_path, str):
        if os.path.isdir(image_path):
            image, filename = load_first_image(image_path)
        else:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            filename = os.path.basename(image_path)
    else:
        image = image_path
        filename = "image"

    # MÉTODO 1: THRESHOLD (original)
    cropped = circular_crop(image, radius=crop_radius)
    _, dark_areas = cv2.threshold(cropped, 85, 255, cv2.THRESH_BINARY_INV)
    _, valid_area = cv2.threshold(cropped, 10, 255, cv2.THRESH_BINARY)
    ventricles_threshold = cv2.bitwise_and(dark_areas, valid_area)

    # Aplicar fechamento morfológico
    k_h = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 5))
    ventricles_threshold_connected = cv2.morphologyEx(ventricles_threshold, cv2.MORPH_CLOSE, k_h)
    k_v = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 10))
    ventricles_threshold_connected = cv2.morphologyEx(ventricles_threshold_connected, cv2.MORPH_CLOSE, k_v)

    contours_threshold, _ = find_contours(ventricles_threshold_connected)
    largest_threshold = get_largest_contour(contours_threshold)

    # MÉTODO 2: CANNY com heurística de área escura
    cropped_canny, edges_canny, contours_canny, ventricles_canny = segment_ventricles_canny(
        image, crop_radius=crop_radius, low_threshold=20, high_threshold=60
    )
    largest_canny = get_largest_contour(contours_canny)

    # Criar visualizações
    steps = []

    # Linha 1: MÉTODO THRESHOLD
    # 1. Original
    step1 = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.putText(step1, "METODO 1: THRESHOLD", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    steps.append(step1)

    # 2. Máscara threshold
    step2 = cv2.cvtColor(ventricles_threshold_connected, cv2.COLOR_GRAY2BGR)
    cv2.putText(step2, "Mascara Threshold", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    steps.append(step2)

    # 3. Contornos threshold
    step3 = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)
    if largest_threshold:
        cv2.drawContours(step3, largest_threshold, -1, (0, 255, 0), 2)
    cv2.putText(step3, f"Contornos ({len(contours_threshold)})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    steps.append(step3)

    # 4. Resultado threshold
    step4 = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if largest_threshold:
        cv2.drawContours(step4, largest_threshold, -1, (0, 255, 0), 3)
    cv2.putText(step4, "Resultado Threshold", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    steps.append(step4)

    # Linha 2: MÉTODO CANNY
    # 5. Original
    step5 = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.putText(step5, "METODO 2: CANNY", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    steps.append(step5)

    # 6. Bordas Canny
    step6 = cv2.cvtColor(edges_canny, cv2.COLOR_GRAY2BGR)
    cv2.putText(step6, "Bordas Canny", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    steps.append(step6)

    # 7. Contornos Canny
    step7 = cv2.cvtColor(cropped_canny, cv2.COLOR_GRAY2BGR)
    if largest_canny:
        cv2.drawContours(step7, largest_canny, -1, (255, 0, 255), 2)
    cv2.putText(step7, f"Contornos ({len(contours_canny)})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    steps.append(step7)

    # 8. Resultado Canny
    step8 = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if largest_canny:
        cv2.drawContours(step8, largest_canny, -1, (255, 0, 255), 3)
    cv2.putText(step8, "Resultado Canny", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    steps.append(step8)

    # Criar grid 2x4
    row1 = np.hstack(steps[0:4])
    row2 = np.hstack(steps[4:8])
    final_image = np.vstack([row1, row2])

    # Salvar
    cv2.imwrite(save_path, final_image)
    print(f"\nComparação de métodos salva em: {save_path}")

    # Estatísticas
    print("\n" + "="*60)
    print("COMPARAÇÃO DE MÉTODOS")
    print("="*60)
    print(f"\nMÉTODO 1 - THRESHOLD:")
    print(f"  Contornos detectados: {len(contours_threshold)}")
    if largest_threshold:
        areas_th = [cv2.contourArea(c) for c in largest_threshold]
        print(f"  Áreas dos ventrículos: {areas_th}")
        print(f"  Área total: {sum(areas_th):.0f} px²")

    print(f"\nMÉTODO 2 - CANNY:")
    print(f"  Contornos detectados: {len(contours_canny)}")
    if largest_canny:
        areas_ca = [cv2.contourArea(c) for c in largest_canny]
        print(f"  Áreas dos ventrículos: {areas_ca}")
        print(f"  Área total: {sum(areas_ca):.0f} px²")
    print("="*60)

    return final_image


def create_processing_flow_image(image_path, dark_threshold=85, brain_threshold=20, crop_radius=250, save_path='fluxo_processamento.png'):
    """
    Cria uma única imagem mostrando todo o fluxo de processamento.

    Args:
        image_path: Caminho da imagem ou diretório
        dark_threshold: Threshold para áreas escuras
        brain_threshold: Threshold para máscara cerebral
        crop_radius: Raio do crop circular
        save_path: Onde salvar a imagem final

    Returns:
        Imagem composta com todas as etapas
    """

    # Carregar imagem
    if isinstance(image_path, str):
        if os.path.isdir(image_path):
            image, filename = load_first_image(image_path)
        else:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            filename = os.path.basename(image_path)
    else:
        image = image_path
        filename = "image"

    # Lista para armazenar cada etapa
    steps = []

    # ETAPA 1: Original
    step1 = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.putText(step1, "1. Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    steps.append(step1)

    # ETAPA 2: Crop Circular
    cropped = circular_crop(image, radius=crop_radius)
    step2 = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)
    # Desenhar circulo verde mostrando o crop
    h, w = image.shape
    cv2.circle(step2, (w//2, h//2), crop_radius, (0, 255, 0), 2)
    cv2.putText(step2, f"2. Crop Circular (r={crop_radius})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    steps.append(step2)

    # ETAPA 3: Binarização
    _, binary = cv2.threshold(cropped, dark_threshold, 255, cv2.THRESH_BINARY)
    step3 = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    cv2.putText(step3, f"3. Binarizacao (th={dark_threshold})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    steps.append(step3)

    # ETAPA 4: Fechamento Morfológico
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    step4 = cv2.cvtColor(binary_closed, cv2.COLOR_GRAY2BGR)
    cv2.putText(step4, "4. Fechamento Morfologico", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    steps.append(step4)

    # ETAPA 5: Máscara de Área Válida
    _, valid_area = cv2.threshold(cropped, 10, 255, cv2.THRESH_BINARY)
    step5 = cv2.cvtColor(valid_area, cv2.COLOR_GRAY2BGR)
    cv2.putText(step5, "5. Area Valida", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    steps.append(step5)

    # ETAPA 6: Áreas Escuras
    _, dark_areas = cv2.threshold(cropped, dark_threshold, 255, cv2.THRESH_BINARY_INV)
    step6 = cv2.cvtColor(dark_areas, cv2.COLOR_GRAY2BGR)
    cv2.putText(step6, "6. Areas Escuras", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    steps.append(step6)

    # ETAPA 7: Ventrículos Segmentados
    ventricles_mask = cv2.bitwise_and(dark_areas, valid_area)
    # Aplicar fechamento morfológico para conectar
    k_h = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 5))
    ventricles_connected = cv2.morphologyEx(ventricles_mask, cv2.MORPH_CLOSE, k_h)
    k_v = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 10))
    ventricles_connected = cv2.morphologyEx(ventricles_connected, cv2.MORPH_CLOSE, k_v)

    step7 = cv2.cvtColor(ventricles_connected, cv2.COLOR_GRAY2BGR)
    cv2.putText(step7, "7. Ventriculos Conectados", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    steps.append(step7)

    # ETAPA 8: Resultado Final (contornos na imagem original)
    contours, _ = find_contours(ventricles_connected)
    largest = get_largest_contour(contours)
    step8 = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if largest:
        cv2.drawContours(step8, largest, -1, (0, 255, 0), 3)
    cv2.putText(step8, "8. Resultado Final", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    steps.append(step8)

    # Criar grid 2x4
    row1 = np.hstack(steps[0:4])
    row2 = np.hstack(steps[4:8])
    final_image = np.vstack([row1, row2])

    # Salvar
    cv2.imwrite(save_path, final_image)
    print(f"\nFluxo de processamento salvo em: {save_path}")

    # Mostrar (não funciona bem no WSL, então comentado)
    # cv2.imshow('Fluxo de Processamento', final_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return final_image

import tkinter as tk
from tkinter import filedialog, ttk, messagebox, Menu
import tkinter.font as tkfont
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os
import joblib
import warnings
import torch

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    print("nibabel não está instalado. Arquivos NIFTI (.nii) desativados.")

from treatment_flow import (
    circular_crop,
    segment_ventricles_canny,
    segment_ventricles_watershed,
    segment_ventricles_otsu,
    segment_ventricles_kmeans,
    segment_ventricles_region_growing,
    get_largest_contour
)


def calculate_descriptors(contours):
    if not contours or len(contours) == 0:
        return {
            'total_area': 0,
            'avg_circularity': 0,
            'eccentricity': 0,
            'total_perimeter': 0,
            'avg_solidity': 0,
            'avg_aspect_ratio': 0
        }

    total_area = sum(cv2.contourArea(c) for c in contours)

    circs = []
    for c in contours:
        area = cv2.contourArea(c)
        perim = cv2.arcLength(c, True)
        if perim > 0:
            circs.append((4 * np.pi * area) / (perim ** 2))
    avg_circ = np.mean(circs) if circs else 0

    largest = max(contours, key=cv2.contourArea)
    eccent = 0
    if len(largest) >= 5:
        try:
            ellipse = cv2.fitEllipse(largest)
            w, h = ellipse[1]
            a = max(w, h) / 2
            b = min(w, h) / 2
            if a > 0:
                eccent = np.sqrt(1 - (b ** 2 / a ** 2))
        except Exception:
            eccent = 0

    total_perim = sum(cv2.arcLength(c, True) for c in contours)

    solids = []
    for c in contours:
        area = cv2.contourArea(c)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solids.append(area / hull_area)
    avg_solid = np.mean(solids) if solids else 0

    aspects = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h > 0:
            aspects.append(w / h)
    avg_aspect = np.mean(aspects) if aspects else 0

    return {
        'total_area': total_area,
        'avg_circularity': avg_circ,
        'eccentricity': eccent,
        'total_perimeter': total_perim,
        'avg_solidity': avg_solid,
        'avg_aspect_ratio': avg_aspect
    }


def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext in ['.nii', '.nii.gz']:
        if not NIBABEL_AVAILABLE:
            raise ValueError("Instale nibabel: pip install nibabel")

        try:
            nii = nib.load(path)
            data = nii.get_fdata()

            if len(data.shape) == 3:
                slice_idx = data.shape[2] // 2
                image = data[:, :, slice_idx]
            else:
                image = data

            if image.max() > image.min():
                image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            else:
                image = np.zeros_like(image, dtype=np.uint8)
            return image
        except Exception as e:
            raise ValueError(f"Erro ao carregar NIFTI: {str(e)}")

    elif ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"OpenCV não conseguiu carregar a imagem: {path}")
        return image

    else:
        raise ValueError(f"Formato não suportado: {ext}")


class VentricleSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Segmentação de Imagens - Alzheimer")
        self.root.geometry("1400x900")

        self.colors = {
            'bg_dark': '#1a1a1a',
            'bg_medium': '#2d2d2d',
            'bg_light': '#3d3d3d',
            'fg_primary': '#ffffff',
            'fg_secondary': '#b0b0b0',
            'accent_blue': '#4a9eff',
            'accent_green': '#4ade80',
            'accent_red': '#ef4444',
            'accent_yellow': '#fbbf24',
            'border': '#404040'
        }

        self.current_image = None
        self.original_image = None
        self.result_image = None
        self.contours = None
        self.descriptors = None
        self.last_model_prediction = None
        self.last_model_probability = None
        self.current_image_path = None
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.dataset_df = None
        self.font_size = tk.IntVar(value=10)
        self.selected_method = tk.StringVar(value="Threshold")

        self.image_list = []
        self.current_index = 0
        self.current_folder = None

        self.crop_radius = tk.IntVar(value=250)
        self.dark_threshold = tk.IntVar(value=85)
        self.canny_low = tk.IntVar(value=20)
        self.canny_high = tk.IntVar(value=60)
        self.kmeans_k = tk.IntVar(value=4)
        self.loaded_models = {}

        self.setup_theme()
        self.create_menu()
        self.create_ui()

    def setup_theme(self):
        self.root.configure(bg=self.colors['bg_dark'])

        style = ttk.Style()
        style.theme_use('clam')

        style.configure(
            'TButton',
            background=self.colors['bg_light'],
            foreground=self.colors['fg_primary'],
            borderwidth=0,
            padding=10
        )
        style.map('TButton', background=[('active', self.colors['accent_blue'])])

        style.configure('TLabel', background=self.colors['bg_dark'], foreground=self.colors['fg_primary'])
        style.configure('TFrame', background=self.colors['bg_dark'])

        style.configure(
            'TLabelframe',
            background=self.colors['bg_dark'],
            foreground=self.colors['fg_primary'],
            borderwidth=2
        )
        style.configure(
            'TLabelframe.Label',
            background=self.colors['bg_dark'],
            foreground=self.colors['accent_blue'],
            font=('Arial', 11, 'bold')
        )

        self.style = style
        self.apply_font_size()

    def create_menu(self):
        menubar = Menu(self.root, bg=self.colors['bg_medium'], fg=self.colors['fg_primary'])
        self.root.config(menu=menubar)

        file_menu = Menu(menubar, tearoff=0, bg=self.colors['bg_medium'], fg=self.colors['fg_primary'])
        menubar.add_cascade(label="Arquivo", menu=file_menu)
        file_menu.add_command(label="Abrir Imagem...", command=self.open_image, accelerator="Ctrl+O")
        file_menu.add_command(label="Salvar Resultado...", command=self.save_result, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="Carregar CSV Dataset...", command=self.load_csv)
        file_menu.add_command(label="Exportar Descritores...", command=self.export_descriptors)
        file_menu.add_separator()
        file_menu.add_command(label="Sair", command=self.root.quit)

        process_menu = Menu(menubar, tearoff=0, bg=self.colors['bg_medium'], fg=self.colors['fg_primary'])
        menubar.add_cascade(label="Processamento", menu=process_menu)
        process_menu.add_command(label="Segmentar Ventrículos", command=self.segment_ventricles, accelerator="Ctrl+P")
        process_menu.add_command(label="Comparar Métodos", command=self.compare_methods)

        models_menu = Menu(process_menu, tearoff=0, bg=self.colors['bg_medium'], fg=self.colors['fg_primary'])
        process_menu.add_cascade(label="Modelos", menu=models_menu)
        self.models_menu = models_menu
        self.populate_models_menu()

        view_menu = Menu(menubar, tearoff=0, bg=self.colors['bg_medium'], fg=self.colors['fg_primary'])
        menubar.add_cascade(label="Visualização", menu=view_menu)
        view_menu.add_command(label="Resetar Zoom", command=self.reset_zoom)
        view_menu.add_command(label="Scatterplots", command=self.show_scatterplots)
        view_menu.add_command(label="Histograma", command=self.show_histogram)

        access_menu = Menu(menubar, tearoff=0, bg=self.colors['bg_medium'], fg=self.colors['fg_primary'])
        menubar.add_cascade(label="Acessibilidade", menu=access_menu)
        access_menu.add_command(label="Aumentar Fonte", command=self.increase_font, accelerator="Ctrl++")
        access_menu.add_command(label="Diminuir Fonte", command=self.decrease_font, accelerator="Ctrl+-")
        access_menu.add_command(label="Resetar Fonte", command=self.reset_font)

        help_menu = Menu(menubar, tearoff=0, bg=self.colors['bg_medium'], fg=self.colors['fg_primary'])
        menubar.add_cascade(label="Ajuda", menu=help_menu)
        help_menu.add_command(label="Sobre", command=self.show_about)

        self.root.bind('<Control-o>', lambda e: self.open_image())
        self.root.bind('<Control-s>', lambda e: self.save_result())
        self.root.bind('<Control-p>', lambda e: self.segment_ventricles())
        self.root.bind('<Control-plus>', lambda e: self.increase_font())
        self.root.bind('<Control-minus>', lambda e: self.decrease_font())
        self.root.bind('<Left>', lambda e: self.prev_image())
        self.root.bind('<Right>', lambda e: self.next_image())

    def get_model_files(self):
        models_dir = os.path.join(os.path.dirname(__file__), 'modelos')
        classifiers, regressors = [], []

        if os.path.isdir(models_dir):
            for filename in sorted(os.listdir(models_dir)):
                if filename.lower().startswith('classifier'):
                    classifiers.append(filename)
                elif filename.lower().startswith('regressor'):
                    regressors.append(filename)

        return classifiers, regressors

    def populate_models_menu(self, show_summary=False):
        if not hasattr(self, 'models_menu'):
            return

        self.models_menu.delete(0, tk.END)

        classifiers, regressors = self.get_model_files()

        classifiers_menu = Menu(self.models_menu, tearoff=0, bg=self.colors['bg_medium'], fg=self.colors['fg_primary'])
        regressors_menu = Menu(self.models_menu, tearoff=0, bg=self.colors['bg_medium'], fg=self.colors['fg_primary'])

        if classifiers:
            for clf in classifiers:
                classifiers_menu.add_command(
                    label=clf,
                    command=lambda name=clf: self.show_model_details("Classificador", name)
                )
        else:
            classifiers_menu.add_command(label="Nenhum classificador", state=tk.DISABLED)

        if regressors:
            for reg in regressors:
                regressors_menu.add_command(
                    label=reg,
                    command=lambda name=reg: self.show_model_details("Regressor", name)
                )
        else:
            regressors_menu.add_command(label="Nenhum regressor", state=tk.DISABLED)

        self.models_menu.add_cascade(label="Classificadores", menu=classifiers_menu)
        self.models_menu.add_cascade(label="Regressores", menu=regressors_menu)
        self.models_menu.add_separator()
        self.models_menu.add_command(
            label="Atualizar lista",
            command=lambda: self.populate_models_menu(show_summary=True)
        )

        if show_summary:
            total = len(classifiers) + len(regressors)
            if total == 0:
                message = "Nenhum modelo na pasta 'modelos'."
            else:
                message = (
                    f"Modelos: {total}\n"
                    f"Classificadores: {len(classifiers)}\n"
                    f"Regressores: {len(regressors)}"
                )
            messagebox.showinfo("Modelos", message)

    def show_model_details(self, model_type, filename):
        models_dir = os.path.join(os.path.dirname(__file__), 'modelos')
        full_path = os.path.join(models_dir, filename)

        if not os.path.exists(full_path):
            messagebox.showerror("Erro", f"Arquivo de modelo não encontrado:\n{full_path}")
            return

        msg = (
            f"{model_type}: {filename}\n\n"
            f"Caminho:\n{full_path}\n\n"
            "Executar este modelo na imagem atual?"
        )

        if messagebox.askyesno("Modelo", msg):
            self.run_model_on_current_image(model_type, full_path)

    def open_models_dialog(self):
        classifiers, regressors = self.get_model_files()

        dialog = tk.Toplevel(self.root)
        dialog.title("Modelos")
        dialog.configure(bg=self.colors['bg_dark'])

        container = tk.Frame(dialog, bg=self.colors['bg_dark'])
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        def add_section(title_text, items, model_type):
            frame = ttk.LabelFrame(container, text=title_text, padding=10)
            frame.pack(fill=tk.X, pady=(0, 10))

            if not items:
                tk.Label(
                    frame,
                    text="Nenhum arquivo encontrado.",
                    bg=self.colors['bg_dark'],
                    fg=self.colors['fg_secondary'],
                    anchor=tk.W
                ).pack(fill=tk.X)
                return

            for name in items:
                btn = tk.Button(
                    frame,
                    text=name,
                    command=lambda n=name: self.show_model_details(model_type, n),
                    bg=self.colors['bg_light'],
                    fg=self.colors['fg_primary'],
                    relief=tk.FLAT,
                    cursor='hand2',
                    anchor=tk.W,
                    padx=10,
                    pady=6
                )
                btn.pack(fill=tk.X, pady=2)

        add_section("Classificadores", classifiers, "Classificador")
        add_section("Regressores", regressors, "Regressor")

        tk.Button(
            container,
            text="Fechar",
            command=dialog.destroy,
            bg=self.colors['accent_red'],
            fg=self.colors['fg_primary'],
            relief=tk.FLAT,
            cursor='hand2',
            padx=12,
            pady=8
        ).pack(pady=(10, 0))

    def adjust_regression_value(self, model_path, value):
        """Aplicar ajuste específico conforme o regressor utilizado."""
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return value

        basename = os.path.basename(model_path).lower()

        if basename == "regressor_metrics_only.pkl":
            return numeric_value / 1000.0
        if basename == "regressor_nn_full.pth":
            return numeric_value + 20.0

        return numeric_value

    def run_model_on_current_image(self, model_type, model_path):
        if self.current_image is None:
            messagebox.showwarning("Aviso", "Carregue uma imagem.")
            return

        basename = os.path.basename(model_path).lower()
        ext = os.path.splitext(basename)[1]
        is_pytorch = ext in [".pth", ".pt"] or "nn" in basename

        try:
            load_warnings = []
            if model_path in self.loaded_models:
                model = self.loaded_models[model_path]
            else:
                if is_pytorch:
                    model = torch.load(
                        model_path,
                        map_location=torch.device("cpu"),
                        weights_only=False
                    )
                    model.eval()
                else:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        model = joblib.load(model_path)
                        load_warnings = [str(w.message) for w in caught]

                self.loaded_models[model_path] = model
        except Exception as e:
            extra_tip = ""
            if "xgboost" in str(e).lower():
                extra_tip = (
                    "\n\nO modelo do XGBoost foi salvo em uma versão antiga. "
                    "Reexporte com Booster.save_model na versão original e carregue novamente."
                )
            messagebox.showerror("Erro", f"Erro ao carregar modelo:\n{str(e)}{extra_tip}")
            return

        if load_warnings:
            formatted = "\n- " + "\n- ".join(load_warnings)
            print("[Avisos modelo]" + formatted)

        if is_pytorch:
            img = self.result_image if self.result_image is not None else self.current_image

            if len(img.shape) == 2:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            target_size = (224, 224)
            img_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_LINEAR)
            img_resized = img_resized.astype(np.float32) / 255.0

            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_norm = (img_resized - mean) / std

            tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0)

            try:
                with torch.no_grad():
                    outputs = model(tensor)

                if outputs.ndim == 2 and outputs.shape[1] == 2:
                    probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
                    pred_label = int(np.argmax(probs))
                    prob_doente = float(probs[1])

                    if pred_label == 1:
                        resultado_texto = "Resultado: paciente com demência."
                    else:
                        resultado_texto = "Resultado: paciente sem demência."

                    proba_text = f"\nProbabilidade classe 1: {prob_doente * 100:.1f}%"

                elif outputs.ndim == 2 and outputs.shape[1] == 1:
                    prob_doente = torch.sigmoid(outputs)[0, 0].item()
                    pred_label = int(prob_doente >= 0.5)

                    if pred_label == 1:
                        resultado_texto = "Resultado: paciente com demência."
                    else:
                        resultado_texto = "Resultado: paciente sem demência."

                    proba_text = f"\nProbabilidade classe 1: {prob_doente * 100:.1f}%"

                else:
                    valor = outputs.squeeze().item()
                    if model_type.lower().startswith("reg"):
                        valor = self.adjust_regression_value(model_path, valor)
                    resultado_texto = f"Saída do modelo: {valor:.4f}"
                    proba_text = ""
            except Exception as e:
                messagebox.showerror("Erro", f"Erro na predição (PyTorch):\n{str(e)}")
                return

            messagebox.showinfo(
                "Resultado do Modelo",
                f"Modelo: {os.path.basename(model_path)}\n"
                f"Tipo: {model_type} (Profundo)\n\n"
                f"{resultado_texto}{proba_text}"
            )
            return

        if self.descriptors is None:
            try:
                self.segment_ventricles()
            except Exception as e:
                messagebox.showerror("Erro", f"Não foi possível segmentar a imagem:\n{str(e)}")
                return

            if self.descriptors is None:
                messagebox.showerror("Erro", "Não foi possível calcular descritores.")
                return

        feature_order = [
            'total_area',
            'avg_circularity',
            'eccentricity',
            'total_perimeter',
            'avg_solidity',
            'avg_aspect_ratio'
        ]

        try:
            x = pd.DataFrame([
                {feature: self.descriptors[feature] for feature in feature_order}
            ])
        except KeyError as e:
            messagebox.showerror(
                "Erro",
                f"Descritor ausente: {str(e)}."
            )
            return

        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                y_pred = model.predict(x)[0]

                proba_text = ""
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(x)[0]
                    if len(proba) > 1:
                        proba_text = f"\nProbabilidade classe 1: {proba[1] * 100:.1f}%"
                predict_warnings = [str(w.message) for w in caught]
        except Exception as e:
            messagebox.showerror("Erro", f"Erro na predição:\n{str(e)}")
            return

        if predict_warnings:
            formatted = "\n- " + "\n- ".join(predict_warnings)
            print("[Avisos predição]" + formatted)

        if model_type.lower().startswith("class"):
            if int(y_pred) == 1:
                resultado_texto = "Resultado: paciente com demência."
            else:
                resultado_texto = "Resultado: paciente sem demência."
        else:
            adjusted_pred = self.adjust_regression_value(model_path, y_pred)
            resultado_texto = f"Valor previsto: {float(adjusted_pred):.3f}"

        if not (self.root and self.root.winfo_exists()):
            return

        messagebox.showinfo(
            "Resultado do Modelo",
            f"Modelo: {os.path.basename(model_path)}\n"
            f"Tipo: {model_type}\n\n"
            f"{resultado_texto}{proba_text}",
            parent=self.root
        )

    def create_ui(self):
        left_container = tk.Frame(self.root, width=320, bg=self.colors['bg_dark'])
        left_container.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=10)
        left_container.pack_propagate(False)

        left_canvas = tk.Canvas(
            left_container,
            bg=self.colors['bg_dark'],
            highlightthickness=0,
            width=300
        )
        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(
            left_container,
            orient=tk.VERTICAL,
            command=left_canvas.yview,
            bg=self.colors['bg_medium'],
            troughcolor=self.colors['bg_light'],
            activebackground=self.colors['accent_blue']
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        left_canvas.configure(yscrollcommand=scrollbar.set)

        left_panel = tk.Frame(left_canvas, bg=self.colors['bg_dark'])
        left_canvas_window_id = left_canvas.create_window((0, 0), window=left_panel, anchor=tk.NW)

        def configure_scroll_region(event=None):
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))

        def configure_canvas_width(event=None):
            canvas_width = event.width if event else left_canvas.winfo_width()
            if left_canvas_window_id:
                left_canvas.itemconfig(left_canvas_window_id, width=canvas_width)

        left_panel.bind('<Configure>', configure_scroll_region)
        left_canvas.bind('<Configure>', configure_canvas_width)

        def on_mousewheel(event):
            left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def on_mousewheel_linux(event):
            if event.num == 4:
                left_canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                left_canvas.yview_scroll(1, "units")

        left_canvas.bind_all("<MouseWheel>", on_mousewheel)
        left_canvas.bind_all("<Button-4>", on_mousewheel_linux)
        left_canvas.bind_all("<Button-5>", on_mousewheel_linux)

        title = tk.Label(
            left_panel,
            text="Controles",
            font=('Arial', 14, 'bold'),
            bg=self.colors['bg_dark'],
            fg=self.colors['accent_blue']
        )
        title.pack(pady=(0, 20))

        load_frame = ttk.LabelFrame(left_panel, text="Imagem", padding=10)
        load_frame.pack(fill=tk.X, pady=(0, 15))

        btn_open = tk.Button(
            load_frame,
            text="Abrir Imagem",
            command=self.open_image,
            bg=self.colors['accent_blue'],
            fg=self.colors['fg_primary'],
            font=('Arial', 10, 'bold'),
            relief=tk.FLAT,
            cursor='hand2',
            padx=20,
            pady=10
        )
        btn_open.pack(fill=tk.X, pady=(0, 5))

        btn_open_folder = tk.Button(
            load_frame,
            text="Abrir Pasta",
            command=self.open_folder,
            bg=self.colors['bg_light'],
            fg=self.colors['fg_primary'],
            font=('Arial', 9),
            relief=tk.FLAT,
            cursor='hand2',
            pady=8
        )
        btn_open_folder.pack(fill=tk.X)

        method_frame = ttk.LabelFrame(left_panel, text="Método de Segmentação", padding=10)
        method_frame.pack(fill=tk.X, pady=(0, 15))

        methods = [
            "Threshold",
            "Canny",
            "Watershed",
            "Otsu",
            "K-Means",
            "Region Growing"
        ]

        for method in methods:
            rb = tk.Radiobutton(
                method_frame,
                text=method,
                variable=self.selected_method,
                value=method,
                bg=self.colors['bg_dark'],
                fg=self.colors['fg_primary'],
                selectcolor=self.colors['bg_medium'],
                activebackground=self.colors['bg_dark'],
                activeforeground=self.colors['accent_blue'],
                font=('Arial', 10)
            )
            rb.pack(anchor=tk.W, pady=2)

        btn_segment = tk.Button(
            method_frame,
            text="Segmentar",
            command=self.segment_ventricles,
            bg=self.colors['accent_green'],
            fg=self.colors['fg_primary'],
            font=('Arial', 10, 'bold'),
            relief=tk.FLAT,
            cursor='hand2',
            padx=20,
            pady=10
        )
        btn_segment.pack(fill=tk.X, pady=(10, 0))

        open_models_callback = getattr(self, "open_models_dialog", None)
        if open_models_callback is None:
            def _missing_models_dialog():
                messagebox.showerror(
                    "Modelos",
                    "Ação de modelos indisponível no momento."
                )

            open_models_callback = _missing_models_dialog
            self.open_models_dialog = open_models_callback

        btn_models = tk.Button(
            method_frame,
            text="Modelos",
            command=open_models_callback,
            bg=self.colors['accent_blue'],
            fg=self.colors['fg_primary'],
            font=('Arial', 10, 'bold'),
            relief=tk.FLAT,
            cursor='hand2',
            padx=20,
            pady=10
        )
        btn_models.pack(fill=tk.X, pady=(8, 0))

        params_frame = ttk.LabelFrame(left_panel, text="Parâmetros", padding=10)
        params_frame.pack(fill=tk.X, pady=(0, 15))

        tk.Label(
            params_frame,
            text="Raio de recorte (px):",
            bg=self.colors['bg_dark'],
            fg=self.colors['fg_secondary'],
            font=('Arial', 9)
        ).pack(anchor=tk.W)

        crop_frame = tk.Frame(params_frame, bg=self.colors['bg_dark'])
        crop_frame.pack(fill=tk.X, pady=(2, 8))

        crop_slider = tk.Scale(
            crop_frame,
            from_=100,
            to=400,
            resolution=10,
            orient=tk.HORIZONTAL,
            variable=self.crop_radius,
            bg=self.colors['bg_medium'],
            fg=self.colors['fg_primary'],
            highlightthickness=0,
            troughcolor=self.colors['bg_light']
        )
        crop_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        crop_value = tk.Label(
            crop_frame,
            textvariable=self.crop_radius,
            bg=self.colors['bg_dark'],
            fg=self.colors['accent_blue'],
            font=('Arial', 9, 'bold'),
            width=4
        )
        crop_value.pack(side=tk.LEFT, padx=(5, 0))

        tk.Label(
            params_frame,
            text="Threshold escuro:",
            bg=self.colors['bg_dark'],
            fg=self.colors['fg_secondary'],
            font=('Arial', 9)
        ).pack(anchor=tk.W)

        dark_frame = tk.Frame(params_frame, bg=self.colors['bg_dark'])
        dark_frame.pack(fill=tk.X, pady=(2, 8))

        dark_slider = tk.Scale(
            dark_frame,
            from_=50,
            to=150,
            resolution=5,
            orient=tk.HORIZONTAL,
            variable=self.dark_threshold,
            bg=self.colors['bg_medium'],
            fg=self.colors['fg_primary'],
            highlightthickness=0,
            troughcolor=self.colors['bg_light']
        )
        dark_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        dark_value = tk.Label(
            dark_frame,
            textvariable=self.dark_threshold,
            bg=self.colors['bg_dark'],
            fg=self.colors['accent_blue'],
            font=('Arial', 9, 'bold'),
            width=4
        )
        dark_value.pack(side=tk.LEFT, padx=(5, 0))

        tk.Label(
            params_frame,
            text="Canny Low/High:",
            bg=self.colors['bg_dark'],
            fg=self.colors['fg_secondary'],
            font=('Arial', 9)
        ).pack(anchor=tk.W)

        canny_frame = tk.Frame(params_frame, bg=self.colors['bg_dark'])
        canny_frame.pack(fill=tk.X, pady=(2, 8))

        tk.Label(
            canny_frame,
            text="Low:",
            bg=self.colors['bg_dark'],
            fg=self.colors['fg_secondary'],
            font=('Arial', 8)
        ).pack(side=tk.LEFT)

        canny_low_slider = tk.Scale(
            canny_frame,
            from_=10,
            to=100,
            resolution=5,
            orient=tk.HORIZONTAL,
            variable=self.canny_low,
            bg=self.colors['bg_medium'],
            fg=self.colors['fg_primary'],
            highlightthickness=0,
            troughcolor=self.colors['bg_light'],
            length=80
        )
        canny_low_slider.pack(side=tk.LEFT, padx=2)

        tk.Label(
            canny_frame,
            text="High:",
            bg=self.colors['bg_dark'],
            fg=self.colors['fg_secondary'],
            font=('Arial', 8)
        ).pack(side=tk.LEFT, padx=(10, 0))

        canny_high_slider = tk.Scale(
            canny_frame,
            from_=30,
            to=150,
            resolution=5,
            orient=tk.HORIZONTAL,
            variable=self.canny_high,
            bg=self.colors['bg_medium'],
            fg=self.colors['fg_primary'],
            highlightthickness=0,
            troughcolor=self.colors['bg_light'],
            length=80
        )
        canny_high_slider.pack(side=tk.LEFT, padx=2)

        tk.Label(
            params_frame,
            text="K-Means clusters:",
            bg=self.colors['bg_dark'],
            fg=self.colors['fg_secondary'],
            font=('Arial', 9)
        ).pack(anchor=tk.W)

        kmeans_frame = tk.Frame(params_frame, bg=self.colors['bg_dark'])
        kmeans_frame.pack(fill=tk.X, pady=(2, 8))

        kmeans_slider = tk.Scale(
            kmeans_frame,
            from_=2,
            to=8,
            resolution=1,
            orient=tk.HORIZONTAL,
            variable=self.kmeans_k,
            bg=self.colors['bg_medium'],
            fg=self.colors['fg_primary'],
            highlightthickness=0,
            troughcolor=self.colors['bg_light']
        )
        kmeans_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        kmeans_value = tk.Label(
            kmeans_frame,
            textvariable=self.kmeans_k,
            bg=self.colors['bg_dark'],
            fg=self.colors['accent_blue'],
            font=('Arial', 9, 'bold'),
            width=4
        )
        kmeans_value.pack(side=tk.LEFT, padx=(5, 0))

        btn_reset_params = tk.Button(
            params_frame,
            text="Resetar parâmetros",
            command=self.reset_parameters,
            bg=self.colors['bg_light'],
            fg=self.colors['fg_secondary'],
            font=('Arial', 8),
            relief=tk.FLAT,
            cursor='hand2',
            pady=5
        )
        btn_reset_params.pack(fill=tk.X, pady=(5, 0))

        zoom_frame = ttk.LabelFrame(left_panel, text="Zoom", padding=10)
        zoom_frame.pack(fill=tk.X, pady=(0, 15))

        tk.Label(
            zoom_frame,
            text="Zoom:",
            bg=self.colors['bg_dark'],
            fg=self.colors['fg_secondary'],
            font=('Arial', 9)
        ).pack(anchor=tk.W)

        zoom_slider = tk.Scale(
            zoom_frame,
            from_=0.25,
            to=5.0,
            resolution=0.25,
            orient=tk.HORIZONTAL,
            command=self.update_zoom,
            bg=self.colors['bg_medium'],
            fg=self.colors['fg_primary'],
            highlightthickness=0,
            troughcolor=self.colors['bg_light']
        )
        zoom_slider.set(1.0)
        zoom_slider.pack(fill=tk.X, pady=(5, 10))

        btn_reset_zoom = tk.Button(
            zoom_frame,
            text="Resetar zoom",
            command=self.reset_zoom,
            bg=self.colors['bg_light'],
            fg=self.colors['fg_primary'],
            font=('Arial', 9),
            relief=tk.FLAT,
            cursor='hand2',
            pady=5
        )
        btn_reset_zoom.pack(fill=tk.X)

        action_frame = ttk.LabelFrame(left_panel, text="Ações", padding=10)
        action_frame.pack(fill=tk.X, pady=(0, 15))

        btn_save = tk.Button(
            action_frame,
            text="Salvar Resultado",
            command=self.save_result,
            bg=self.colors['bg_light'],
            fg=self.colors['fg_primary'],
            font=('Arial', 9),
            relief=tk.FLAT,
            cursor='hand2',
            pady=8
        )
        btn_save.pack(fill=tk.X, pady=(0, 5))

        btn_export = tk.Button(
            action_frame,
            text="Exportar Descritores",
            command=self.export_descriptors,
            bg=self.colors['bg_light'],
            fg=self.colors['fg_primary'],
            font=('Arial', 9),
            relief=tk.FLAT,
            cursor='hand2',
            pady=8
        )
        btn_export.pack(fill=tk.X, pady=(0, 5))

        btn_scatter = tk.Button(
            action_frame,
            text="Scatterplots",
            command=self.show_scatterplots,
            bg=self.colors['bg_light'],
            fg=self.colors['fg_primary'],
            font=('Arial', 9),
            relief=tk.FLAT,
            cursor='hand2',
            pady=8
        )
        btn_scatter.pack(fill=tk.X)

        center_panel = ttk.Frame(self.root)
        center_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        img_title = tk.Label(
            center_panel,
            text="Imagem",
            font=('Arial', 14, 'bold'),
            bg=self.colors['bg_dark'],
            fg=self.colors['accent_blue']
        )
        img_title.pack(pady=(0, 10))

        canvas_frame = tk.Frame(center_panel, bg=self.colors['bg_medium'], relief=tk.SOLID, borderwidth=2)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(
            canvas_frame,
            bg='#0a0a0a',
            highlightthickness=0
        )
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        self.canvas.bind('<ButtonPress-1>', self.on_mouse_down)
        self.canvas.bind('<B1-Motion>', self.on_mouse_drag)
        self.canvas.bind('<MouseWheel>', self.on_mouse_wheel)

        self.info_label = tk.Label(
            center_panel,
            text="Nenhuma imagem carregada",
            font=('Arial', 9),
            bg=self.colors['bg_dark'],
            fg=self.colors['fg_secondary']
        )
        self.info_label.pack(pady=(5, 0))

        nav_frame = tk.Frame(center_panel, bg=self.colors['bg_dark'])
        nav_frame.pack(pady=10)

        self.btn_prev = tk.Button(
            nav_frame,
            text="Anterior",
            command=self.prev_image,
            bg=self.colors['bg_light'],
            fg=self.colors['fg_primary'],
            font=('Arial', 10, 'bold'),
            relief=tk.FLAT,
            cursor='hand2',
            padx=20,
            pady=8,
            state=tk.DISABLED
        )
        self.btn_prev.pack(side=tk.LEFT, padx=5)

        self.nav_label = tk.Label(
            nav_frame,
            text="",
            bg=self.colors['bg_dark'],
            fg=self.colors['fg_primary'],
            font=('Arial', 10, 'bold')
        )
        self.nav_label.pack(side=tk.LEFT, padx=20)

        self.btn_next = tk.Button(
            nav_frame,
            text="Próxima",
            command=self.next_image,
            bg=self.colors['bg_light'],
            fg=self.colors['fg_primary'],
            font=('Arial', 10, 'bold'),
            relief=tk.FLAT,
            cursor='hand2',
            padx=20,
            pady=8,
            state=tk.DISABLED
        )
        self.btn_next.pack(side=tk.LEFT, padx=5)

        right_panel = ttk.Frame(self.root, width=350)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10, pady=10)
        right_panel.pack_propagate(False)

        desc_title = tk.Label(
            right_panel,
            text="Descritores",
            font=('Arial', 14, 'bold'),
            bg=self.colors['bg_dark'],
            fg=self.colors['accent_blue']
        )
        desc_title.pack(pady=(0, 20))

        desc_frame = ttk.LabelFrame(right_panel, text="Valores", padding=10)
        desc_frame.pack(fill=tk.BOTH, pady=(0, 15))

        desc_inner = tk.Frame(desc_frame, bg=self.colors['bg_medium'], padx=15, pady=15)
        desc_inner.pack(fill=tk.BOTH, expand=True)

        self.desc_value_labels = {}

        descriptors_info = [
            ("Área Total:", "total_area", "px²"),
            ("Circularidade:", "avg_circularity", ""),
            ("Excentricidade:", "eccentricity", ""),
            ("Perímetro Total:", "total_perimeter", "px"),
            ("Solidez:", "avg_solidity", ""),
            ("Aspect Ratio:", "avg_aspect_ratio", "")
        ]

        for i, (label_text, key, unit) in enumerate(descriptors_info):
            row_frame = tk.Frame(desc_inner, bg=self.colors['bg_medium'])
            row_frame.pack(fill=tk.X, pady=4)

            label = tk.Label(
                row_frame,
                text=label_text,
                bg=self.colors['bg_medium'],
                fg=self.colors['fg_secondary'],
                font=('Arial', 10),
                anchor=tk.W,
                width=18
            )
            label.pack(side=tk.LEFT)

            value_label = tk.Label(
                row_frame,
                text="---",
                bg=self.colors['bg_light'],
                fg=self.colors['accent_blue'],
                font=('Courier', 10, 'bold'),
                anchor=tk.E,
                width=12,
                relief=tk.FLAT,
                padx=8,
                pady=4
            )
            value_label.pack(side=tk.LEFT, padx=(5, 0))

            if unit:
                unit_label = tk.Label(
                    row_frame,
                    text=unit,
                    bg=self.colors['bg_medium'],
                    fg=self.colors['fg_secondary'],
                    font=('Arial', 9),
                    anchor=tk.W,
                    width=5
                )
                unit_label.pack(side=tk.LEFT, padx=(5, 0))

            self.desc_value_labels[key] = value_label

        graph_frame = ttk.LabelFrame(right_panel, text="Gráfico", padding=10)
        graph_frame.pack(fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(3.5, 2.5), facecolor=self.colors['bg_medium'])
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(self.colors['bg_light'])
        self.ax.tick_params(colors=self.colors['fg_secondary'])

        self.canvas_graph = FigureCanvasTkAgg(self.fig, graph_frame)
        self.canvas_graph.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.ax.text(
            0.5,
            0.5,
            'Aguardando dados...',
            ha='center',
            va='center',
            color=self.colors['fg_secondary'],
            transform=self.ax.transAxes
        )
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas_graph.draw()

    def open_image(self):
        path = filedialog.askopenfilename(
            title="Abrir Imagem",
            filetypes=[
                ("Imagens", "*.png *.jpg *.jpeg *.bmp"),
                ("NIFTI", "*.nii *.nii.gz"),
                ("Todos", "*.*")
            ]
        )

        if not path:
            return

        self.image_list = []
        self.current_index = 0
        self.current_folder = None
        self.update_nav_buttons()

        try:
            image = load_image(path)

            if image is None:
                raise ValueError("Imagem carregada é None")

            self.original_image = image.copy()
            self.current_image = image
            self.current_image_path = path
            self.result_image = None
            self.contours = None
            self.descriptors = None
            self.last_model_prediction = None
            self.last_model_probability = None

            self.display_image(image)
            self.update_info(f"Imagem: {os.path.basename(path)} | Shape: {image.shape}")

            for key in self.desc_value_labels:
                self.desc_value_labels[key].config(text="---")

        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar imagem:\n{str(e)}")

    def open_folder(self):
        folder = filedialog.askdirectory(title="Selecionar Pasta com Imagens")

        if not folder:
            return

        try:
            supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.nii', '.nii.gz')
            all_files = os.listdir(folder)

            self.image_list = []
            for filename in sorted(all_files):
                if filename.lower().endswith(supported_formats):
                    full_path = os.path.join(folder, filename)
                    self.image_list.append(full_path)

            if not self.image_list:
                messagebox.showwarning("Aviso", "Nenhuma imagem encontrada.")
                return

            self.current_folder = folder
            self.current_index = 0
            self.load_current_image()
            self.update_nav_buttons()

            messagebox.showinfo("Info", f"Pasta carregada.\nImagens: {len(self.image_list)}")

        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao abrir pasta:\n{str(e)}")

    def load_current_image(self):
        if not self.image_list or self.current_index >= len(self.image_list):
            return

        path = self.image_list[self.current_index]

        try:
            image = load_image(path)

            if image is None:
                raise ValueError("Imagem carregada é None")

            self.original_image = image.copy()
            self.current_image = image
            self.current_image_path = path
            self.result_image = None
            self.contours = None
            self.descriptors = None
            self.last_model_prediction = None
            self.last_model_probability = None

            self.display_image(image)
            filename = os.path.basename(path)
            self.update_info(f"{filename} | {self.current_index + 1}/{len(self.image_list)} | Shape: {image.shape}")

            for key in self.desc_value_labels:
                self.desc_value_labels[key].config(text="---")

        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar imagem:\n{str(e)}")

    def prev_image(self):
        if not self.image_list:
            return

        self.current_index = (self.current_index - 1) % len(self.image_list)
        self.load_current_image()
        self.update_nav_buttons()

    def next_image(self):
        if not self.image_list:
            return

        self.current_index = (self.current_index + 1) % len(self.image_list)
        self.load_current_image()
        self.update_nav_buttons()

    def update_nav_buttons(self):
        if self.image_list and len(self.image_list) > 1:
            self.btn_prev.config(state=tk.NORMAL)
            self.btn_next.config(state=tk.NORMAL)
            self.nav_label.config(text=f"{self.current_index + 1} / {len(self.image_list)}")
        else:
            self.btn_prev.config(state=tk.DISABLED)
            self.btn_next.config(state=tk.DISABLED)
            self.nav_label.config(text="")

    def segment_ventricles(self):
        if self.current_image is None:
            messagebox.showwarning("Aviso", "Carregue uma imagem.")
            return

        method = self.selected_method.get()

        crop_r = self.crop_radius.get()
        dark_th = self.dark_threshold.get()
        canny_l = self.canny_low.get()
        canny_h = self.canny_high.get()
        k_clusters = self.kmeans_k.get()

        try:
            if method == "Threshold":
                cropped = circular_crop(self.original_image, radius=crop_r)
                _, dark_areas = cv2.threshold(cropped, dark_th, 255, cv2.THRESH_BINARY_INV)
                _, valid_area = cv2.threshold(cropped, 10, 255, cv2.THRESH_BINARY)
                ventricles = cv2.bitwise_and(dark_areas, valid_area)
                k_h = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 5))
                ventricles = cv2.morphologyEx(ventricles, cv2.MORPH_CLOSE, k_h)
                k_v = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 10))
                ventricles = cv2.morphologyEx(ventricles, cv2.MORPH_CLOSE, k_v)
                contours_found, _ = cv2.findContours(ventricles, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                largest = get_largest_contour(contours_found)

            elif method == "Canny":
                _, _, largest, _ = segment_ventricles_canny(
                    self.original_image,
                    crop_radius=crop_r,
                    low_threshold=canny_l,
                    high_threshold=canny_h
                )

            elif method == "Watershed":
                _, _, largest, _ = segment_ventricles_watershed(self.original_image, crop_radius=crop_r)

            elif method == "Otsu":
                _, _, largest, _ = segment_ventricles_otsu(self.original_image, crop_radius=crop_r)

            elif method == "K-Means":
                _, _, largest, _ = segment_ventricles_kmeans(
                    self.original_image,
                    crop_radius=crop_r,
                    k=k_clusters
                )

            elif method == "Region Growing":
                _, _, largest, _ = segment_ventricles_region_growing(self.original_image, crop_radius=crop_r)

            result = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2BGR)
            if largest:
                cv2.drawContours(result, largest, -1, (0, 255, 0), 2)

            self.result_image = result
            self.contours = largest
            self.descriptors = calculate_descriptors(largest if largest else [])

            self.run_default_model_prediction()

            self.display_image(result)
            self.update_descriptors()
            self.update_graph()

            messagebox.showinfo("Info", f"Segmentação concluída ({method}).")

        except Exception as e:
            messagebox.showerror("Erro", f"Erro na segmentação:\n{str(e)}")

    def save_result(self):
        if self.result_image is None:
            messagebox.showwarning("Aviso", "Nenhum resultado para salvar.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")]
        )

        if path:
            cv2.imwrite(path, self.result_image)
            messagebox.showinfo("Info", f"Resultado salvo em:\n{path}")

    def display_image(self, image):
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image_rgb.shape[:2]
        new_w = int(w * self.zoom_level)
        new_h = int(h * self.zoom_level)

        if new_w > 0 and new_h > 0:
            image_resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            image_resized = image_rgb

        pil_image = Image.fromarray(image_resized)
        self.photo = ImageTk.PhotoImage(pil_image)

        self.canvas.delete("all")
        self.canvas.create_image(self.pan_x, self.pan_y, anchor=tk.NW, image=self.photo)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

    def update_info(self, text):
        self.info_label.config(text=text)

    def update_descriptors(self):
        if self.descriptors is None:
            for key in self.desc_value_labels:
                self.desc_value_labels[key].config(text="---")
            return

        self.desc_value_labels['total_area'].config(
            text=f"{self.descriptors['total_area']:.2f}"
        )
        self.desc_value_labels['avg_circularity'].config(
            text=f"{self.descriptors['avg_circularity']:.4f}"
        )
        self.desc_value_labels['eccentricity'].config(
            text=f"{self.descriptors['eccentricity']:.4f}"
        )
        self.desc_value_labels['total_perimeter'].config(
            text=f"{self.descriptors['total_perimeter']:.2f}"
        )
        self.desc_value_labels['avg_solidity'].config(
            text=f"{self.descriptors['avg_solidity']:.4f}"
        )
        self.desc_value_labels['avg_aspect_ratio'].config(
            text=f"{self.descriptors['avg_aspect_ratio']:.4f}"
        )

    def update_graph(self):
        if self.descriptors is None:
            return

        self.ax.clear()

        labels = ['Área\n(x100)', 'Circ.', 'Excent.', 'Perím.\n(x10)', 'Solid.', 'Aspect']
        values = [
            self.descriptors['total_area'] / 100,
            self.descriptors['avg_circularity'],
            self.descriptors['eccentricity'],
            self.descriptors['total_perimeter'] / 10,
            self.descriptors['avg_solidity'],
            self.descriptors['avg_aspect_ratio']
        ]

        colors = [
            self.colors['accent_blue'],
            self.colors['accent_green'],
            self.colors['accent_yellow'],
            self.colors['accent_red'],
            '#a78bfa',
            '#fb923c'
        ]

        self.ax.bar(labels, values, color=colors, alpha=0.8)

        self.ax.set_facecolor(self.colors['bg_light'])
        self.ax.tick_params(colors=self.colors['fg_secondary'], labelsize=8)
        self.ax.spines['bottom'].set_color(self.colors['border'])
        self.ax.spines['top'].set_color(self.colors['border'])
        self.ax.spines['left'].set_color(self.colors['border'])
        self.ax.spines['right'].set_color(self.colors['border'])
        self.ax.set_ylabel('Valor', color=self.colors['fg_secondary'], fontsize=9)
        self.ax.set_title('Descritores', color=self.colors['fg_primary'], fontsize=10, pad=10)

        self.fig.tight_layout()
        self.canvas_graph.draw()

    def update_zoom(self, value):
        self.zoom_level = float(value)
        if self.current_image is not None:
            img = self.result_image if self.result_image is not None else self.current_image
            self.display_image(img)

    def reset_zoom(self):
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        if self.current_image is not None:
            img = self.result_image if self.result_image is not None else self.current_image
            self.display_image(img)

    def on_mouse_down(self, event):
        self.drag_start_x = event.x
        self.drag_start_y = event.y

    def on_mouse_drag(self, event):
        dx = event.x - self.drag_start_x
        dy = event.y - self.drag_start_y

        self.pan_x += dx
        self.pan_y += dy

        self.drag_start_x = event.x
        self.drag_start_y = event.y

        if self.current_image is not None:
            img = self.result_image if self.result_image is not None else self.current_image
            self.display_image(img)

    def on_mouse_wheel(self, event):
        if event.delta > 0:
            self.zoom_level = min(5.0, self.zoom_level + 0.25)
        else:
            self.zoom_level = max(0.25, self.zoom_level - 0.25)

        if self.current_image is not None:
            img = self.result_image if self.result_image is not None else self.current_image
            self.display_image(img)

    def apply_font_size(self):
        size = self.font_size.get()

        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(size=size)
        text_font = tkfont.nametofont("TkTextFont")
        text_font.configure(size=size)
        menu_font = tkfont.nametofont("TkMenuFont")
        menu_font.configure(size=size)

        self.style.configure('.', font=('Arial', size))
        self.style.configure('Treeview', font=('Arial', size))
        self.style.configure('Treeview.Heading', font=('Arial', size, 'bold'))

        self.root.option_add('*Font', ('Arial', size))

    def increase_font(self):
        self.font_size.set(min(20, self.font_size.get() + 2))
        self.apply_font_size()
        messagebox.showinfo("Acessibilidade", f"Fonte: {self.font_size.get()}pt")

    def decrease_font(self):
        self.font_size.set(max(8, self.font_size.get() - 2))
        self.apply_font_size()
        messagebox.showinfo("Acessibilidade", f"Fonte: {self.font_size.get()}pt")

    def reset_font(self):
        self.font_size.set(10)
        self.apply_font_size()
        messagebox.showinfo("Acessibilidade", "Fonte resetada para 10pt")

    def reset_parameters(self):
        self.crop_radius.set(250)
        self.dark_threshold.set(85)
        self.canny_low.set(20)
        self.canny_high.set(60)
        self.kmeans_k.set(4)
        messagebox.showinfo("Parâmetros", "Parâmetros padrão definidos.")

    def load_csv(self):
        path = filedialog.askopenfilename(
            title="Carregar CSV",
            filetypes=[("CSV", "*.csv"), ("Todos", "*.*")]
        )

        if not path:
            return

        try:
            self.dataset_df = pd.read_csv(path, delimiter=';')
            messagebox.showinfo("Info", f"CSV carregado.\nLinhas: {len(self.dataset_df)}")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar CSV:\n{str(e)}")

    def export_descriptors(self):
        if self.descriptors is None:
            messagebox.showwarning("Aviso", "Nenhum descritor calculado.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")]
        )

        if not path:
            return

        image_name = os.path.basename(self.current_image_path) if self.current_image_path else ""

        self.descriptors["Filename"] = os.path.basename(path)
        self.descriptors["Imagem"] = image_name
        self.descriptors["Modelo_predicao"] = self.last_model_prediction if self.last_model_prediction is not None else ""
        self.descriptors["Modelo_proba_classe1"] = self.last_model_probability if self.last_model_probability is not None else ""

        colunas = [
            "Filename",
            "Imagem",
            "total_area",
            "avg_circularity",
            "eccentricity",
            "total_perimeter",
            "avg_solidity",
            "avg_aspect_ratio",
            "Modelo_predicao",
            "Modelo_proba_classe1"
        ]

        df = pd.DataFrame([self.descriptors], columns=colunas)

        arquivo_existe = os.path.exists(path)

        df.to_csv(
            path,
            mode='a',
            header=not arquivo_existe,
            sep=';',
            index=False
        )

        messagebox.showinfo("Info", f"Descritores salvos em:\n{path}")

    def run_default_model_prediction(self):
        """
        Executa automaticamente o classificador baseado apenas em métricas
        sempre que a segmentação é concluída e armazena o resultado para
        exportação.
        """

        if self.descriptors is None:
            return

        models_dir = os.path.join(os.path.dirname(__file__), 'modelos')
        default_model = 'classifier_metrics_only.pkl'
        model_path = os.path.join(models_dir, default_model)

        if not os.path.exists(model_path):
            self.last_model_prediction = None
            self.last_model_probability = None
            return

        try:
            if model_path in self.loaded_models:
                model = self.loaded_models[model_path]
            else:
                model = joblib.load(model_path)
                self.loaded_models[model_path] = model
        except Exception:
            self.last_model_prediction = None
            self.last_model_probability = None
            return

        feature_order = [
            'total_area',
            'avg_circularity',
            'eccentricity',
            'total_perimeter',
            'avg_solidity',
            'avg_aspect_ratio'
        ]

        try:
            x = np.array([[self.descriptors[f] for f in feature_order]], dtype=float)
        except KeyError:
            self.last_model_prediction = None
            self.last_model_probability = None
            return

        try:
            y_pred = model.predict(x)[0]
            proba = None
            if hasattr(model, "predict_proba"):
                probas = model.predict_proba(x)[0]
                if len(probas) > 1:
                    proba = float(probas[1])

            self.last_model_prediction = int(y_pred) if str(y_pred).isdigit() else y_pred
            self.last_model_probability = proba if proba is not None else ""
        except Exception:
            self.last_model_prediction = None
            self.last_model_probability = None

    def show_scatterplots(self):
        if self.dataset_df is None:
            messagebox.showwarning("Aviso", "Carregue um CSV.")
            return

        win = tk.Toplevel(self.root)
        win.title("Scatterplots")
        win.geometry("1200x800")
        win.configure(bg=self.colors['bg_dark'])

        if 'CDR' not in self.dataset_df.columns and 'Group' not in self.dataset_df.columns:
            messagebox.showwarning("Aviso", "CSV sem coluna de classe (CDR ou Group).")
            return

        class_col = 'CDR' if 'CDR' in self.dataset_df.columns else 'Group'
        color_map = {
            'Converted': 'black',
            'Nondemented': 'blue',
            'Demented': 'red',
            0: 'blue',
            0.5: 'red',
            1: 'red',
            2: 'red'
        }

        colors = [color_map.get(c, 'gray') for c in self.dataset_df[class_col]]

        features = [
            'total_area',
            'avg_circularity',
            'eccentricity',
            'total_perimeter',
            'avg_solidity',
            'avg_aspect_ratio'
        ]

        available_features = [f for f in features if f in self.dataset_df.columns]

        if len(available_features) < 2:
            messagebox.showwarning("Aviso", "Poucas features para scatterplot.")
            return

        fig = Figure(figsize=(12, 8), facecolor=self.colors['bg_dark'])

        n_features = len(available_features)
        plot_idx = 1
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if plot_idx > 6:
                    break

                ax = fig.add_subplot(2, 3, plot_idx)
                ax.set_facecolor(self.colors['bg_light'])

                ax.scatter(
                    self.dataset_df[available_features[i]],
                    self.dataset_df[available_features[j]],
                    c=colors,
                    alpha=0.6,
                    s=30
                )

                ax.set_xlabel(available_features[i], color=self.colors['fg_primary'])
                ax.set_ylabel(available_features[j], color=self.colors['fg_primary'])
                ax.tick_params(colors=self.colors['fg_secondary'])
                ax.spines['bottom'].set_color(self.colors['border'])
                ax.spines['top'].set_color(self.colors['border'])
                ax.spines['left'].set_color(self.colors['border'])
                ax.spines['right'].set_color(self.colors['border'])

                plot_idx += 1

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, win)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()

    def show_histogram(self):
        if self.current_image is None:
            messagebox.showwarning("Aviso", "Carregue uma imagem.")
            return

        win = tk.Toplevel(self.root)
        win.title("Histograma")
        win.geometry("600x400")
        win.configure(bg=self.colors['bg_dark'])

        fig = Figure(figsize=(6, 4), facecolor=self.colors['bg_dark'])
        ax = fig.add_subplot(111)
        ax.set_facecolor(self.colors['bg_light'])

        hist = cv2.calcHist([self.current_image], [0], None, [256], [0, 256])
        ax.plot(hist, color=self.colors['accent_blue'])
        ax.set_xlabel('Intensidade', color=self.colors['fg_primary'])
        ax.set_ylabel('Frequência', color=self.colors['fg_primary'])
        ax.set_title('Histograma', color=self.colors['fg_primary'])
        ax.tick_params(colors=self.colors['fg_secondary'])
        ax.spines['bottom'].set_color(self.colors['border'])
        ax.spines['top'].set_color(self.colors['border'])
        ax.spines['left'].set_color(self.colors['border'])
        ax.spines['right'].set_color(self.colors['border'])

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, win)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()

    def compare_methods(self):
        if self.current_image is None:
            messagebox.showwarning("Aviso", "Carregue uma imagem.")
            return

        messagebox.showinfo("Info", "Comparação de métodos ainda em desenvolvimento.")

    def show_about(self):
        about_text = (
            "Trabalho de Processamento de Imagens - Doença de Alzheimer\n\n"
            "Funcionalidades simples:\n"
            "- Abrir imagens (Nifti, PNG, JPG)\n"
            "- Segmentar ventrículos\n"
            "- Calcular descritores\n"
            "- Gerar gráficos e usar modelos\n"
        )
        messagebox.showinfo("Sobre", about_text)


if __name__ == "__main__":
    root = tk.Tk()
    app = VentricleSegmentationApp(root)
    root.mainloop()

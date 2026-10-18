"""Funções compartilhadas pelos scripts de análise."""
import glob
import json
import os

import cv2
import numpy as np

from src.perspective import output_size

# Folha A4 (ISO 216): 210 mm x 297 mm.
A4_WIDTH_MM = 210.0
CORNER_ORDER = ("tl", "tr", "br", "bl")


def list_frames(folder: str) -> list[str]:
    """Índices dos quadros que têm o par <quadro>_original.png / <quadro>_adjusted.png na pasta."""
    frames = []
    for path in glob.glob(os.path.join(folder, "*_original.png")):
        index = os.path.basename(path).split("_")[0]
        if os.path.exists(os.path.join(folder, f"{index}_adjusted.png")):
            frames.append(index)
    if not frames:
        raise FileNotFoundError(f"Nenhum par <quadro>_original.png / <quadro>_adjusted.png em '{folder}'.")
    return sorted(frames, key=int)


def load_original(folder: str, index: str) -> np.ndarray:
    return cv2.imread(os.path.join(folder, f"{index}_original.png"))


def load_adjusted(folder: str, index: str) -> np.ndarray:
    return cv2.imread(os.path.join(folder, f"{index}_adjusted.png"))


def dst_rect(width: int, height: int) -> np.ndarray:
    """Vértices do retângulo de destino, na mesma ordem e convenção de src/perspective.py."""
    return np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])


def a4_size(frame_shape=(1080, 1920), height: int = 1080) -> tuple[int, int]:
    """Tamanho (largura, altura) da saída A4 usada pela aplicação: 764 x 1080 para altura 1080."""
    return output_size(frame_shape, "a4", height)


def load_corners(path: str) -> dict[str, np.ndarray]:
    """Lê o JSON de cantos ({quadro: {"src": [[x, y] x 4], ...}}) e devolve {quadro: array 4x2 (tl, tr, br, bl)}."""
    data = json.load(open(path))
    return {index: np.float64(item["src"]) for index, item in sorted(data.items(), key=lambda kv: int(kv[0]))}


def save_json(data, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)

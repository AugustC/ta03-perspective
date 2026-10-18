"""Recupera os cantos usados em cada exemplo e verifica a retificação.

Para cada par <quadro>_original.png / <quadro>_adjusted.png, estima por ECC
(Evangelidis e Psarakis, 2008), modelo MOTION_HOMOGRAPHY, a homografia que leva o
quadro original ao retificado. Os cantos do documento são os vértices da saída
levados de volta ao quadro original. Em seguida o quadro é retificado de novo com
esses cantos e comparado à saída da aplicação (PSNR e erro absoluto médio).

Uso (na raiz do repositório):
    python -m analise.recover_corners outs/datasheet001.avi --out resultados/corners.json
"""
import argparse

import cv2
import numpy as np

from analise.common import dst_rect, list_frames, load_adjusted, load_original, save_json


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Ordena 4 pontos como tl, tr, br, bl (soma e diferença das coordenadas)."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], np.float32)


def initial_corners(img: np.ndarray) -> np.ndarray:
    """Estimativa inicial da folha: região de matiz rosada (H > 135 no OpenCV) contra o fundo azulado."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 0] > 135) & (hsv[..., 2] > 150)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((41, 41), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((31, 31), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    for eps in (0.01, 0.02, 0.03, 0.05):
        approx = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True)
        if len(approx) == 4:
            return order_corners(approx)
    return order_corners(cv2.boxPoints(cv2.minAreaRect(contour)))


def recover(original: np.ndarray, adjusted: np.ndarray) -> np.ndarray:
    """Cantos (tl, tr, br, bl) no quadro original que geram `adjusted` por homografia."""
    h, w = adjusted.shape[:2]
    dst = dst_rect(w, h)
    # homografia saída -> original, inicializada pela detecção da folha e refinada em pirâmide
    m_inv = cv2.getPerspectiveTransform(dst, initial_corners(original)).astype(np.float32)
    g_org = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g_adj = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-7)
    for scale in (0.25, 0.5, 1.0):
        S = np.diag([scale, scale, 1]).astype(np.float32)
        m_s = (S @ m_inv @ np.linalg.inv(S)).astype(np.float32)
        a = cv2.resize(g_adj, None, fx=scale, fy=scale)
        o = cv2.resize(g_org, None, fx=scale, fy=scale)
        _, m_s = cv2.findTransformECC(a, o, m_s, cv2.MOTION_HOMOGRAPHY, criteria, None, 5)
        m_inv = np.linalg.inv(S) @ m_s @ S
        m_inv /= m_inv[2, 2]
    return cv2.perspectiveTransform(dst[None], m_inv.astype(np.float64))[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", help="Pasta com os pares <quadro>_original.png / <quadro>_adjusted.png.")
    parser.add_argument("--out", default="resultados/corners.json", help="Arquivo JSON de saída.")
    args = parser.parse_args()

    result = {}
    for index in list_frames(args.folder):
        original, adjusted = load_original(args.folder, index), load_adjusted(args.folder, index)
        src = recover(original, adjusted)
        h, w = adjusted.shape[:2]
        redo = cv2.warpPerspective(original, cv2.getPerspectiveTransform(src.astype(np.float32), dst_rect(w, h)), (w, h))
        diff = np.abs(redo.astype(int) - adjusted.astype(int))
        result[index] = dict(src=src.tolist(), psnr=float(cv2.PSNR(redo, adjusted)), mae=float(diff.mean()))
        print(f"quadro {index:>4}: cantos {np.round(src, 1).tolist()}  "
              f"PSNR {result[index]['psnr']:.1f} dB  erro médio {result[index]['mae']:.2f}")
    save_json(result, args.out)
    print(f"salvo em {args.out}")


if __name__ == "__main__":
    main()

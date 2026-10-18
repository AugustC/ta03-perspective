"""Geometria do documento em cada quadro e estimativas da proporção da folha.

Para cada quadro: área do quadrilátero (% do quadro), maior desvio dos ângulos
internos em relação a 90 graus, razões entre lados opostos e a razão altura/largura
estimada de três formas: pelos lados médios, pelos lados máximos e pelo método de
Zhang e He (2007), que estima também a distância focal supondo pixels quadrados e
ponto principal no centro da imagem.

Uso (na raiz do repositório):
    python -m analise.geometry resultados/corners.json --frame-size 1920 1080 --out resultados/geometry.json
"""
import argparse

import cv2
import numpy as np

from analise.common import load_corners, save_json


def interior_angle(p, q, r) -> float:
    """Ângulo (graus) no vértice q."""
    a, b = p - q, r - q
    return float(np.degrees(np.arccos(a @ b / np.linalg.norm(a) / np.linalg.norm(b))))


def zhang_he(tl, tr, br, bl, u0: float, v0: float):
    """Distância focal (px) e razão altura/largura do retângulo real; (None, None) quando f não é determinado.

    Z. Zhang e L.-W. He, "Whiteboard scanning and image enhancement", Digital Signal Processing 17(2), 2007.
    m1..m4 = cantos (0,0), (w,0), (0,h), (w,h) do retângulo, em coordenadas homogêneas.
    """
    m1, m2, m3, m4 = (np.array([*p, 1.0]) for p in (tl, tr, bl, br))
    k2 = np.cross(m1, m4) @ m3 / (np.cross(m2, m4) @ m3)
    k3 = np.cross(m1, m4) @ m2 / (np.cross(m3, m4) @ m2)
    n2, n3 = k2 * m2 - m1, k3 * m3 - m1
    denom = n2[2] * n3[2]
    if abs(denom) < 1e-12:      # um par de lados opostos paralelo na imagem: ponto de fuga no infinito
        return None, None
    f2 = -((n2[0] * n3[0] - (n2[0] * n3[2] + n2[2] * n3[0]) * u0 + n2[2] * n3[2] * u0 ** 2) +
           (n2[1] * n3[1] - (n2[1] * n3[2] + n2[2] * n3[1]) * v0 + n2[2] * n3[2] * v0 ** 2)) / denom
    if not np.isfinite(f2) or f2 <= 0:
        return None, None
    f = float(np.sqrt(f2))
    a_inv = np.linalg.inv(np.array([[f, 0, u0], [0, f, v0], [0, 0, 1]]))
    w_over_h = np.sqrt((n2 @ a_inv.T @ a_inv @ n2) / (n3 @ a_inv.T @ a_inv @ n3))
    return f, float(1 / w_over_h)


def measure(src: np.ndarray, frame_w: int, frame_h: int) -> dict:
    tl, tr, br, bl = src
    top, right = np.linalg.norm(tr - tl), np.linalg.norm(br - tr)
    bottom, left = np.linalg.norm(br - bl), np.linalg.norm(bl - tl)
    angles = [interior_angle(bl, tl, tr), interior_angle(tl, tr, br), interior_angle(tr, br, bl), interior_angle(br, bl, tl)]
    area = float(cv2.contourArea(src.astype(np.float32)))
    focal, hw_zh = zhang_he(tl, tr, br, bl, (frame_w - 1) / 2, (frame_h - 1) / 2)
    return dict(
        sides=dict(top=top, right=right, bottom=bottom, left=left), angles=angles,
        area_pct=100 * area / (frame_w * frame_h),
        max_angle_dev=max(abs(a - 90) for a in angles),
        bottom_top=bottom / top, right_left=right / left,
        hw_mean_sides=((left + right) / 2) / ((top + bottom) / 2),
        hw_max_sides=max(left, right) / max(top, bottom),
        hw_zhang_he=hw_zh, focal_px=focal,
        mm_per_px=210.0 / ((top + bottom) / 2),        # largura A4 / largura média na imagem
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corners", help="JSON de cantos (recover_corners ou corners_from_xml).")
    parser.add_argument("--frame-size", nargs=2, type=int, default=[1920, 1080], metavar=("W", "H"))
    parser.add_argument("--out", default="resultados/geometry.json")
    args = parser.parse_args()
    fw, fh = args.frame_size

    rows = {index: measure(src, fw, fh) for index, src in load_corners(args.corners).items()}
    print(f"{'quadro':>6} {'área %':>7} {'desvio':>7} {'inf/sup':>8} {'dir/esq':>8} {'h/w méd':>8} {'h/w máx':>8} {'h/w ZH':>7} {'f (px)':>7}")
    for index, r in rows.items():
        zh = f"{r['hw_zhang_he']:.3f}" if r['hw_zhang_he'] else "--"
        f = f"{r['focal_px']:.0f}" if r['focal_px'] else "--"
        print(f"{index:>6} {r['area_pct']:7.1f} {r['max_angle_dev']:7.1f} {r['bottom_top']:8.2f} {r['right_left']:8.2f} "
              f"{r['hw_mean_sides']:8.3f} {r['hw_max_sides']:8.3f} {zh:>7} {f:>7}")
    valid_zh = [r['hw_zhang_he'] for r in rows.values() if r['hw_zhang_he']]
    summary = dict(
        mean_area_pct=float(np.mean([r['area_pct'] for r in rows.values()])),
        mean_max_angle_dev=float(np.mean([r['max_angle_dev'] for r in rows.values()])),
        mean_bottom_top=float(np.mean([r['bottom_top'] for r in rows.values()])),
        mean_right_left=float(np.mean([r['right_left'] for r in rows.values()])),
        mean_hw_mean_sides=float(np.mean([r['hw_mean_sides'] for r in rows.values()])),
        mean_hw_max_sides=float(np.mean([r['hw_max_sides'] for r in rows.values()])),
        mean_hw_zhang_he=float(np.mean(valid_zh)) if valid_zh else None,
    )
    print(f"{'média':>6} {summary['mean_area_pct']:7.1f} {summary['mean_max_angle_dev']:7.1f} {summary['mean_bottom_top']:8.2f} "
          f"{summary['mean_right_left']:8.2f} {summary['mean_hw_mean_sides']:8.3f} {summary['mean_hw_max_sides']:8.3f} "
          f"{summary['mean_hw_zhang_he']:7.3f}   (A4 real: {np.sqrt(2):.3f})")
    save_json(dict(frames=rows, summary=summary), args.out)
    print(f"salvo em {args.out}")


if __name__ == "__main__":
    main()

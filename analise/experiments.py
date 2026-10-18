"""Experimentos do artigo: proporção da saída, ordem e erro nos cantos, nitidez e tempo.

Gera as figuras do artigo e um JSON com os números das Tabelas 2, 3 e 4:
  - quadros_cantos.jpg       quadros com o quadrilátero anotado (Figura 1)
  - 15_original.jpg, 15_ta03.jpg      exemplo da saída da versão inicial (Figura 2)
  - proporcao_{ta03,lados,a4}.jpg     três tamanhos de destino (Figura 3)
  - ordem_{correta,espelhada,cruzada}.jpg   ordem dos cantos (Figura 4)
  - nitidez_recortes.jpg     mesmo trecho da página nos quadros (Figura 5)

Uso (na raiz do repositório):
    python -m analise.experiments outs/datasheet001.avi resultados/corners.json --figs resultados/figuras --out resultados/experiments.json
"""
import argparse
import os
import time

import cv2
import numpy as np

from analise.common import A4_WIDTH_MM, a4_size, dst_rect, load_adjusted, load_corners, load_original, save_json

EXAMPLE = "15"   # quadro usado nas figuras de exemplo


def warp(img, src, width, height, flags=cv2.INTER_LINEAR):
    m = cv2.getPerspectiveTransform(np.float32(src), dst_rect(width, height))
    return cv2.warpPerspective(img, m, (width, height), flags=flags)


def save_jpg(path, img, quality=90):
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])


def hstack_with_gap(tiles, gap):
    sep = np.full((tiles[0].shape[0], gap, 3), 255, np.uint8)
    out = tiles[0]
    for t in tiles[1:]:
        out = np.hstack([out, sep, t])
    return out


def fig_corners(folder, corners, figs):
    tiles = []
    for index, src in corners.items():
        img = load_original(folder, index).copy()
        cv2.polylines(img, [src.astype(np.int32)], True, (40, 200, 40), 4, cv2.LINE_AA)
        for k, p in enumerate(src):
            color = (0, 0, 255) if k == 0 else (0, 140, 255)   # vermelho = tl
            cv2.circle(img, tuple(int(v) for v in p), 12, color, -1, cv2.LINE_AA)
        crop = img[60:900, 560:1400]
        for color, thick in (((255, 255, 255), 8), ((30, 30, 30), 3)):
            cv2.putText(crop, f"#{index}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, thick, cv2.LINE_AA)
        tiles.append(cv2.resize(crop, (420, 420), interpolation=cv2.INTER_AREA))
    save_jpg(os.path.join(figs, "quadros_cantos.jpg"), hstack_with_gap(tiles, 8))


def fig_example(folder, figs):
    for name, img in (("original", load_original(folder, EXAMPLE)), ("ta03", load_adjusted(folder, EXAMPLE))):
        save_jpg(os.path.join(figs, f"{EXAMPLE}_{name}.jpg"), cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA), 88)


def fig_proportion(folder, corners, figs):
    img, src = load_original(folder, EXAMPLE), corners[EXAMPLE]
    tl, tr, br, bl = src
    w_max = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    h_max = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    a4w, a4h = a4_size(img.shape)
    outs = {"ta03": warp(img, src, img.shape[1], img.shape[0]),
            "lados": warp(img, src, w_max, h_max),
            "a4": warp(img, src, a4w, a4h)}
    for name, out in outs.items():
        h = 700
        w = int(round(out.shape[1] * h / out.shape[0]))
        save_jpg(os.path.join(figs, f"proporcao_{name}.jpg"), cv2.resize(out, (w, h), interpolation=cv2.INTER_AREA))
    return dict(max_sides_size=[w_max, h_max], a4_size=[a4w, a4h])


def fig_order(folder, corners, figs):
    img, (tl, tr, br, bl) = load_original(folder, EXAMPLE), corners[EXAMPLE]
    a4w, a4h = a4_size(img.shape)
    orders = {"correta": [tl, tr, br, bl], "espelhada": [tr, tl, bl, br], "cruzada": [tl, br, tr, bl]}
    for name, src in orders.items():
        out = warp(img, src, a4w, a4h)
        save_jpg(os.path.join(figs, f"ordem_{name}.jpg"), cv2.resize(out, (382, 540), interpolation=cv2.INTER_AREA))


def sharpness(folder, corners, figs):
    """Variância do Laplaciano (Pech-Pacheco et al., 2000) e razão Ex/Ey das energias de gradiente (Sobel)."""
    result, tiles = {}, []
    for index, src in corners.items():
        img = load_original(folder, index)
        a4w, a4h = a4_size(img.shape)
        page = warp(img, src, a4w, a4h)
        g = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)[30:a4h - 30, 30:a4w - 30]
        gx, gy = cv2.Sobel(g, cv2.CV_64F, 1, 0), cv2.Sobel(g, cv2.CV_64F, 0, 1)
        result[index] = dict(laplacian_var=float(cv2.Laplacian(g, cv2.CV_64F).var()),
                             ex_ey=float((gx ** 2).mean() / (gy ** 2).mean()))
        crop = page[55:415, 70:430].copy()
        cv2.rectangle(crop, (0, 312), (88, 360), (255, 255, 255), -1)
        cv2.putText(crop, f"#{index}", (6, 348), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2, cv2.LINE_AA)
        tiles.append(crop)
        print(f"quadro {index:>4}: variância do Laplaciano {result[index]['laplacian_var']:6.1f}  Ex/Ey {result[index]['ex_ey']:.2f}")
    save_jpg(os.path.join(figs, "nitidez_recortes.jpg"), hstack_with_gap(tiles, 6), 92)
    return result


def sensitivity(corners, frame_shape, sigmas=(0.5, 1, 2, 5, 10), trials=200, seed=0):
    """Deslocamento (mm do papel A4) de uma grade de 25 x 35 pontos da página quando os cantos têm ruído gaussiano."""
    rng = np.random.default_rng(seed)
    a4w, a4h = a4_size(frame_shape)
    gx, gy = np.meshgrid(np.linspace(0, a4w - 1, 25), np.linspace(0, a4h - 1, 35))
    grid = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float64)[None]
    mm_per_px = A4_WIDTH_MM / (a4w - 1)
    result = {}
    for sigma in sigmas:
        means, maxima = [], []
        for src in corners.values():
            m_true = cv2.getPerspectiveTransform(src.astype(np.float32), dst_rect(a4w, a4h))
            pts_img = cv2.perspectiveTransform(grid, np.linalg.inv(m_true))
            for _ in range(trials):
                noisy = (src + rng.normal(0, sigma, src.shape)).astype(np.float32)
                d = np.linalg.norm(cv2.perspectiveTransform(pts_img, cv2.getPerspectiveTransform(noisy, dst_rect(a4w, a4h)))[0]
                                   - grid[0], axis=1) * mm_per_px
                means.append(d.mean()); maxima.append(d.max())
        result[str(sigma)] = dict(mean_mm=float(np.mean(means)), max_p95_mm=float(np.percentile(maxima, 95)))
        print(f"sigma {sigma:>4} px: deslocamento médio {result[str(sigma)]['mean_mm']:.2f} mm, máximo (p95) {result[str(sigma)]['max_p95_mm']:.2f} mm")
    return result


def timing(folder, corners, runs=60, warmup=10):
    """Mediana do tempo de getPerspectiveTransform + warpPerspective por interpolação e tamanho de saída."""
    img, src = load_original(folder, EXAMPLE), np.float32(corners[EXAMPLE])
    sizes = {"frame": (img.shape[1], img.shape[0]), "a4": a4_size(img.shape)}
    interps = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC, "lanczos4": cv2.INTER_LANCZOS4}
    result = {}
    for iname, flag in interps.items():
        for sname, (w, h) in sizes.items():
            times = []
            for _ in range(runs):
                t0 = time.perf_counter()
                cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst_rect(w, h)), (w, h), flags=flag)
                times.append(time.perf_counter() - t0)
            result[f"{iname}_{w}x{h}_ms"] = float(np.median(times[warmup:]) * 1000)
            print(f"{iname:>9} {w}x{h}: {result[f'{iname}_{w}x{h}_ms']:.2f} ms")
    times = []
    for _ in range(2000):
        t0 = time.perf_counter(); cv2.getPerspectiveTransform(src, dst_rect(*sizes["frame"])); times.append(time.perf_counter() - t0)
    result["getPerspectiveTransform_us"] = float(np.median(times) * 1e6)
    result["opencv"], result["threads"] = cv2.__version__, cv2.getNumThreads()
    print(f"getPerspectiveTransform: {result['getPerspectiveTransform_us']:.1f} us (OpenCV {cv2.__version__}, {cv2.getNumThreads()} threads)")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", help="Pasta com os pares <quadro>_original.png / <quadro>_adjusted.png.")
    parser.add_argument("corners", help="JSON de cantos (recover_corners ou corners_from_xml).")
    parser.add_argument("--figs", default="resultados/figuras", help="Pasta das figuras.")
    parser.add_argument("--out", default="resultados/experiments.json")
    parser.add_argument("--skip-timing", action="store_true", help="Não mede o tempo (mais rápido).")
    args = parser.parse_args()

    os.makedirs(args.figs, exist_ok=True)
    corners = load_corners(args.corners)
    frame_shape = load_original(args.folder, next(iter(corners))).shape
    fig_corners(args.folder, corners, args.figs)
    fig_example(args.folder, args.figs)
    fig_order(args.folder, corners, args.figs)
    result = dict(proportion=fig_proportion(args.folder, corners, args.figs),
                  sharpness=sharpness(args.folder, corners, args.figs),
                  sensitivity=sensitivity(corners, frame_shape))
    if not args.skip_timing:
        result["timing"] = timing(args.folder, corners)
    save_json(result, args.out)
    print(f"figuras em {args.figs}; números em {args.out}")


if __name__ == "__main__":
    main()

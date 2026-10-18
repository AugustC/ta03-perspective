"""Gera o JSON de cantos a partir do XML de anotação do SmartDoc.

Alternativa a recover_corners.py quando se tem o arquivo de anotação: lê os cantos
dos quadros que têm par original/retificado na pasta e grava no mesmo formato.

Uso (na raiz do repositório):
    python -m analise.corners_from_xml outs/datasheet001.avi dados/datasheet001.xml --out resultados/corners.json
"""
import argparse

from analise.common import CORNER_ORDER, list_frames, save_json
from src.io import extract_ground_truth


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", help="Pasta com os pares <quadro>_original.png / <quadro>_adjusted.png.")
    parser.add_argument("xml", help="Arquivo XML de anotação do vídeo.")
    parser.add_argument("--out", default="resultados/corners.json", help="Arquivo JSON de saída.")
    args = parser.parse_args()

    ground_truth = extract_ground_truth(args.xml)
    result = {}
    for index in list_frames(args.folder):
        if index not in ground_truth:
            print(f"quadro {index}: sem anotação no XML, ignorado")
            continue
        result[index] = dict(src=[list(ground_truth[index][name]) for name in CORNER_ORDER])
    save_json(result, args.out)
    print(f"{len(result)} quadros salvos em {args.out}")


if __name__ == "__main__":
    main()

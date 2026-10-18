# ta03-perspective

Retificação de documentos filmados com transformação de perspectiva (homografia), em Python com OpenCV.
A aplicação lê um vídeo do conjunto SmartDoc e o arquivo XML com os cantos anotados do documento,
sorteia quadros anotados e salva, para cada um, a imagem original e a retificada.

## Instalação

Requer Python 3.10+.

```bash
uv sync                      # ou: pip install opencv-python
```

## Uso

```bash
python main.py <video.avi> <anotacao.xml> <pasta_saida> [opções]
```

| Opção | Padrão | Descrição |
|---|---|---|
| `--n_sample N` | 5 | Número de quadros anotados sorteados. |
| `--seed S` | aleatória | Semente do sorteio, para repetir a mesma escolha de quadros. |
| `--ratio R` | `a4` | Proporção da saída: `a4` (altura/largura = √2), `frame` (mesmo tamanho do quadro do vídeo, o comportamento anterior, que estica a página) ou um número (altura/largura). |
| `--height H` | 1080 | Altura da imagem retificada, em pixels (com `a4`, a saída tem 764 × 1080). |

Exemplo:

```bash
python main.py dados/datasheet001.avi dados/datasheet001.xml outs --seed 0
```

As imagens são salvas em `<pasta_saida>/<nome_do_video>/<quadro>_original.png` e `<quadro>_adjusted.png`.
Quadros sem anotação, marcados como `rejected` ou com cantos que não formam um quadrilátero válido
na ordem `tl, tr, br, bl` são ignorados (com aviso).

Os exemplos em `outs/datasheet001.avi` foram gerados com a versão anterior, equivalente a `--ratio frame`.

## Testes

```bash
python -m unittest -v
```

Os testes geram um vídeo e um XML sintéticos; não é preciso baixar o SmartDoc.

## Dados

Esse projeto utiliza os dados da competição SmartDoc da conferência ICDAR'15.

Citação:
 Jean-Christophe Burie, Joseph Chazalon, Mickaël Coustaty, Sébastien Eskenazi, Muhammad Muzzamil Luqman, Maroua Mehri, Nibal Nayef, Jean-Marc OGIER, Sophea Prum and Marçal Rusinol: “ICDAR2015 Competition on Smartphone Document Capture and OCR (SmartDoc)”, In 13th International Conference on Document Analysis and Recognition (ICDAR), 2015.

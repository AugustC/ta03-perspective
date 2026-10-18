# Análises do artigo

Scripts que geram os números e as figuras das Seções 3.3 e 4 do artigo
(*Retificação de Documentos Filmados por Transformação de Perspectiva*).
Eles usam os pares `<quadro>_original.png` / `<quadro>_adjusted.png` de
`outs/datasheet001.avi`, gerados pela versão inicial da aplicação (equivalente a `--ratio frame`).

Rode na raiz do repositório, nesta ordem:

```bash
# 1. cantos de cada quadro + verificação da retificação (Seção 4.1)
python -m analise.recover_corners outs/datasheet001.avi --out resultados/corners.json
#    ou, com o XML de anotação do vídeo:
python -m analise.corners_from_xml outs/datasheet001.avi dados/datasheet001.xml --out resultados/corners.json

# 2. geometria e proporção estimada (Tabela 1)
python -m analise.geometry resultados/corners.json --out resultados/geometry.json

# 3. figuras, sensibilidade, nitidez e tempo (Figuras 1-5, Tabelas 2-4)
python -m analise.experiments outs/datasheet001.avi resultados/corners.json --figs resultados/figuras --out resultados/experiments.json
```

| Script | O que faz | Onde aparece no artigo |
|---|---|---|
| `recover_corners.py` | Estima por ECC a homografia entre o quadro original e o retificado, recupera os cantos e retifica de novo para comparar (PSNR, erro médio). | Seção 4.1 |
| `corners_from_xml.py` | Lê os mesmos cantos direto do XML do SmartDoc (alternativa ao passo anterior). | — |
| `geometry.py` | Área, desvio dos ângulos, razões entre lados e razão altura/largura pelos lados e pelo método de Zhang e He (2007). | Tabela 1, Seção 4.3 |
| `experiments.py` | Figuras de exemplo, proporção da saída, ordem dos cantos, ruído nos cantos (Monte Carlo), nitidez e tempo de execução. | Figuras 1-5, Tabelas 2-4 |

O sorteio do Monte Carlo usa semente fixa (0), então a Tabela 2 sai igual em toda execução.
Os tempos dependem da máquina e variam entre execuções; use `--skip-timing` para pular essa parte.

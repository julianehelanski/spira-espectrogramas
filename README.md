# gerar_espectrogramas_spira

Script de geração das formas de onda, espectrogramas mel e diagrama de convolução CNN utilizados no Capítulo 4 da dissertação **"A rede que Marcelo construiu"** (Helanski, 2026), a partir de gravações do dataset público do projeto SPIRA (IME-USP / C4AI-USP).

---

## Figuras geradas

| Arquivo | Descrição |
|---|---|
| `spira_waveform_controle.png` | Forma de onda bruta — grupo controle, sem eixos |
| `spira_waveform_paciente.png` | Forma de onda bruta — grupo paciente, sem eixos |
| `spira_controle_sem_legenda.png` | Espectrograma mel — grupo controle, sem eixos |
| `spira_controle_com_eixos.png` | Espectrograma mel — grupo controle, com eixos e barra de cor |
| `spira_paciente_sem_legenda.png` | Espectrograma mel — grupo paciente, sem eixos |
| `spira_paciente_com_eixos.png` | Espectrograma mel — grupo paciente, com eixos e barra de cor |
| `spira_cnn_diagrama.png` | Diagrama de convolução CNN — espectrograma mel do paciente, filtro 3×3 e mapa de ativação |

Os pares sem eixos / com eixos reproduzem o gesto analítico de tornar visível a cadeia de transformação do sinal acústico em inscrição circulável, nos termos de Latour (2001). O diagrama CNN torna visível a operação que o algoritmo realiza sobre essa inscrição: a varredura do filtro convolucional sobre a imagem do som.

---

## Figuras geradas a partir de `AUD-20260506-WA0053.opus`

Subprojeto em [`projeto-daniela-feriani/`](projeto-daniela-feriani/). Aplicando as mesmas funções e os mesmos parâmetros SPIRA a uma gravação adicional (63,14 s, áudio de WhatsApp em formato `.opus`, decodificado e reamostrado para 16 kHz pelo `librosa.load`), o repositório inclui três figuras que cobrem cada etapa da cadeia de transformação:

### 1. `projeto-daniela-feriani/AUD-20260506-WA0053_waveform.png` — Forma de onda

**O que é.** Gráfico do sinal de áudio bruto: cada pixel horizontal corresponde a uma amostra de pressão do ar capturada pelo microfone (uma a cada 1/16.000 de segundo); o eixo vertical é a amplitude instantânea normalizada. Linha branca sobre fundo preto, sem eixos nem legenda.

**O que faz.** Mostra o polo mais material da cadeia, antes de qualquer decomposição em frequências. Permite ler de relance a estrutura temporal da fala: blocos densos de oscilação (sílabas vocalizadas), zonas finas próximas à linha de zero (pausas e silêncios) e picos isolados (oclusivas, batidas, ruídos transientes). Não revela conteúdo espectral — para isso é preciso o espectrograma.

**Como é gerada.** `salvar_waveform()` em `gerar_espectrogramas_spira.py:122`. Plot direto de `y` × tempo via `matplotlib`, com eixos desligados e fundo preto para uniformidade visual com os espectrogramas sem legenda.

### 2. `projeto-daniela-feriani/AUD-20260506-WA0053_mel_sem_legenda.png` — Espectrograma mel sem eixos

**O que é.** Imagem bidimensional 128 × T da matriz mel-espectral em decibéis, sem eixos, ticks nem barra de cor. Eixo horizontal: tempo (frames). Eixo vertical: 128 bandas perceptuais Mel entre 0 e 8.000 Hz. Cor (`magma`): energia em dB, do mais escuro (silêncio) ao mais claro (energia alta).

**O que faz.** Apresenta a "imagem do som" como ela entra no classificador da CNN: textura espectral pura, antes da nomeação das coordenadas. Concentrações horizontais brilhantes nas bandas inferiores correspondem a formantes vocálicos; estrias verticais marcam consoantes e ataques transientes; faixas pretas verticais marcam pausas respiratórias. É essa matriz — e não o áudio — que a rede neural "vê".

**Como é gerada.** `salvar_sem_eixos()` em `gerar_espectrogramas_spira.py:159`. Pipeline: `librosa.feature.melspectrogram(y, sr=16000, n_mels=128, fmax=8000)` → `librosa.power_to_db(S, ref=np.max)` → `librosa.display.specshow` com `x_axis=None, y_axis=None`.

### 3. `projeto-daniela-feriani/AUD-20260506-WA0053_mel_com_eixos.png` — Espectrograma mel com eixos e barra de cor

**O que é.** Mesma matriz da figura 2, mas com eixos calibrados — Tempo (s) no horizontal, Frequência (Mel) no vertical — e barra de cor lateral em decibéis (`+0 dB` no topo, energia máxima de referência; valores negativos indicando atenuação relativa). Fundo branco, título identificando arquivo e parâmetros.

**O que faz.** Torna a inscrição legível e comparável: as coordenadas nomeadas convertem a textura em objeto que pode circular entre laboratórios, ser citado em publicações e justapor-se a outros espectrogramas. É a forma canônica de apresentação em artigos do projeto SPIRA e equivalentes.

**Como é gerada.** `salvar_com_eixos()` em `gerar_espectrogramas_spira.py:187`. Mesmo pipeline da figura 2, com `x_axis="time", y_axis="mel", fmax=8000` e `fig.colorbar` formatada em `%+2.0f dB`.

### Reprodução

```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from gerar_espectrogramas_spira import (
    carregar_audio, salvar_waveform, calcular_espectrograma,
    salvar_sem_eixos, salvar_com_eixos,
)
base = 'projeto-daniela-feriani/AUD-20260506-WA0053'
y, sr = carregar_audio(f'{base}.opus')
salvar_waveform(y, sr, f'{base}_waveform.png')
S = calcular_espectrograma(y, sr)
salvar_sem_eixos(S, sr, f'{base}_mel_sem_legenda.png')
salvar_com_eixos(S, sr, f'{base}_mel_com_eixos.png',
                 titulo='Espectrograma Mel — AUD-20260506-WA0053 | 128 coeficientes, 16 kHz')
"
```

---

## Parâmetros técnicos

Reproduzem o padrão descrito nos artigos do projeto SPIRA:

| Parâmetro | Valor |
|---|---|
| Taxa de amostragem (`sr`) | 16.000 Hz |
| Coeficientes Mel (`n_mels`) | 128 |
| Frequência máxima (`fmax`) | 8.000 Hz |
| Colormap | `magma` |

Referências:

- Casanova Gris et al. (2021). Towards a COVID-19 respiratory insufficiency detection system based on speech. *Findings of ACL-IJCNLP 2021*, p. 617–628. Disponível em: https://aclanthology.org/2021.findings-acl.55. Acesso em: 13 mar. 2026.
- Gauy et al. (2024). Discriminant analysis for respiratory insufficiency using deep learning models and transfer learning. *arXiv:2511.14939*. Disponível em: https://arxiv.org/abs/2511.14939. Acesso em: 13 mar. 2026.
- Younesi, A. et al. (2024). A comprehensive survey of convolutions in deep learning: applications, challenges, and future trends. *arXiv:2402.15490*. Disponível em: https://arxiv.org/abs/2402.15490. Acesso em: 17 mar. 2026.

---

## Dataset público SPIRA

| Recurso | URL | Licença |
|---|---|---|
| Repositório principal (código e dataset, ACL 2021) | https://github.com/SPIRA-COVID19/SPIRA-ACL2021 | CC BY-SA 4.0 |
| Organização geral do projeto | https://github.com/spirabr | — |
| Áudios de fala (pacientes e controles) | https://drive.google.com/file/d/1Bv0d3uwBB-52MBmtN2A_qNoaBIxUkN9y/view | CC BY-SA 4.0 |
| Ruídos de enfermaria hospitalar | https://drive.google.com/file/d/1zNwkye2FhV5LOVh3OfdqgPKzmYS7LeCM/view | CC BY-SA 4.0 |

O dataset não está incluído neste repositório. Para reproduzir as figuras, faça o download dos arquivos de áudio pelos links acima e informe os caminhos via argumentos de linha de comando (ver seção Uso).

---

## Instalação

```bash
pip install -r requirements.txt
```

Versões mínimas testadas: Python 3.10, librosa 0.10, matplotlib 3.7, numpy 1.24, scipy 1.10.

---

## Uso

```bash
# gera todas as figuras (formas de onda, espectrogramas e diagrama CNN)
python gerar_espectrogramas_spira.py \
    --controle caminho/para/controle.wav \
    --paciente caminho/para/PTT-20200511-WA0018.wav \
    --saida    figuras/cap.4/
```

```bash
# gera apenas o diagrama CNN, com espectrograma simulado (sem arquivos .wav)
python gerar_espectrogramas_spira.py --apenas-cnn --saida figuras/cap.4/
```

Se os argumentos forem omitidos, o script busca os arquivos no diretório corrente com os nomes padrão `spira_controle.wav` e `PTT-20200511-WA0018.wav`, e salva as figuras no diretório corrente.

---

## Cadeia de transformação

O script expõe as quatro etapas da cadeia de transformação do sinal acústico em inscrição e classificação:

1. **Forma de onda** (`salvar_waveform`): o sinal bruto como sequência de amostras de pressão do ar. Polo mais material da cadeia, antes de qualquer decomposição espectral.
2. **Espectrograma mel sem eixos** (`salvar_sem_eixos`): a matriz bidimensional 128 × T visualizada como textura espectral, antes da nomeação das coordenadas.
3. **Espectrograma mel com eixos** (`salvar_com_eixos`): o mesmo objeto com tempo (s), frequência (Mel) e amplitude (dB) nomeados, no estado em que pode circular entre laboratórios.
4. **Diagrama CNN** (`gerar_diagrama_cnn`): três painéis em sequência — espectrograma mel do paciente com a janela do filtro 3×3 demarcada sobre uma região de pausa respiratória; filtro com pesos aprendidos; mapa de ativação resultante, com as regiões de pausa destacadas. Quando um arquivo `.wav` de paciente é fornecido, as pausas são detectadas automaticamente a partir da energia espectral no interior da fala (excluídos os 15 primeiros e últimos frames), selecionando os três centros de grupo com menor energia. Quando nenhum arquivo é fornecido (`--apenas-cnn`), o espectrograma é simulado com os parâmetros técnicos do projeto.

---

## Nota metodológica

Este script foi produzido com auxílio do modelo de linguagem Claude Sonnet 4.6 (Anthropic, 2025–2026) em sessão de trabalho conduzida em 12 de março de 2026. A interpretação analítica das imagens geradas é da pesquisadora.

O uso do modelo de linguagem como ferramenta de pesquisa é discutido no Apêndice da dissertação ("Nota sobre o uso de modelo de linguagem como ferramenta de pesquisa"), onde a recursividade entre objeto de estudo e ferramenta metodológica é tratada como dado reflexivo do campo.

---

## Citação

Se este script for utilizado em trabalhos acadêmicos, cite a dissertação de origem:

> HELANSKI, Juliane. *A rede que Marcelo construiu*: etnografia do projeto SPIRA e do Centro de Inteligência Artificial da USP. Tese (Doutorado) — [Programa de Pós-Graduação], Universidade Estadual de Campinas, 2026.

e o dataset público do SPIRA:

> CASANOVA GRIS, Edresson et al. Towards a COVID-19 respiratory insufficiency detection system based on speech. In: *Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021*. Stroudsburg: ACL, 2021. p. 617–628.

---

## Licença

O código deste repositório é disponibilizado sob licença MIT. As gravações do dataset SPIRA estão sob licença CC BY-SA 4.0 (ver links acima).

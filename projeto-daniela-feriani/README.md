# projeto-daniela-feriani

Subprojeto independente: análise visual de uma gravação de áudio em três representações (forma de onda + dois espectrogramas mel). Auto-contido — não depende de nada fora desta pasta.

---

## Arquivos

| Arquivo | Tipo | Descrição |
|---|---|---|
| `AUD-20260506-WA0053.opus` | áudio | Gravação original (63,14 s, `.opus` do WhatsApp) — formato compacto, mas não tocado pelo player do GitHub |
| `AUD-20260506-WA0053.wav` | áudio | Mesma gravação convertida para WAV 16 kHz mono — **toca direto no navegador** ao clicar no arquivo no GitHub |
| `gerar_imagens.py` | código | Script que gera as três imagens a partir do áudio |
| `gerar_video.py` | código | Script que gera o vídeo MP4 da forma de onda com áudio sincronizado |
| `requirements.txt` | dependências | `librosa`, `matplotlib`, `numpy` (+ `ffmpeg` no sistema para o vídeo) |
| `AUD-20260506-WA0053_waveform.png` | imagem | Forma de onda bruta — linha branca sobre fundo preto, sem eixos |
| `AUD-20260506-WA0053_mel_sem_legenda.png` | imagem | Espectrograma mel 128 × T em dB, sem eixos nem barra de cor |
| `AUD-20260506-WA0053_mel_com_eixos.png` | imagem | Espectrograma mel com Tempo (s), Frequência (Mel) e barra de dB |
| `AUD-20260506-WA0053_video.mp4` | vídeo | Forma de onda estilo Wavepad: parte tocada em ciano, parte futura em cinza, playhead branco; áudio sincronizado (63 s, 30 fps) |

---

## O que cada gráfico é e faz

### Forma de onda (`_waveform.png`)

Plot do sinal de áudio bruto: cada ponto horizontal é uma amostra de pressão do ar capturada pelo microfone (uma a cada 1/16.000 s); o eixo vertical é a amplitude instantânea normalizada. Mostra o polo mais material da cadeia, antes de qualquer decomposição em frequências. Permite ler a estrutura temporal da fala — blocos densos de oscilação são sílabas vocalizadas, zonas finas próximas ao zero são pausas, picos isolados são oclusivas ou ruídos transientes. Não revela conteúdo espectral.

### Espectrograma mel sem eixos (`_mel_sem_legenda.png`)

Imagem bidimensional 128 × T da matriz mel-espectral em decibéis. Eixo horizontal: tempo em frames. Eixo vertical: 128 bandas perceptuais Mel entre 0 e 8.000 Hz. Cor (`magma`): energia em dB. É a "imagem do som" como ela entraria em uma CNN: textura espectral pura, antes da nomeação das coordenadas. Concentrações horizontais brilhantes nas bandas inferiores são formantes vocálicos; estrias verticais marcam consoantes e ataques transientes; faixas pretas verticais marcam pausas.

### Espectrograma mel com eixos (`_mel_com_eixos.png`)

Mesma matriz da figura anterior, mas com eixos calibrados — Tempo (s) no horizontal, Frequência (Mel) no vertical — e barra de cor em decibéis (`+0 dB` no topo, valores negativos abaixo indicando atenuação relativa). É a forma canônica de circulação científica do espectrograma: as coordenadas nomeadas convertem a textura em objeto comparável e citável.

---

## Como reproduzir

A partir desta pasta:

```bash
pip install -r requirements.txt
python gerar_imagens.py
```

Ou apontando para outro áudio:

```bash
python gerar_imagens.py --audio caminho/para/outro.opus --saida ./
```

O script aceita qualquer formato suportado por `librosa.load` (wav, mp3, flac, opus, ogg, m4a, ...). Para `.opus` em particular, é necessário ter `ffmpeg` instalado no sistema.

### Vídeo (forma de onda + áudio sincronizado)

```bash
# requer ffmpeg no PATH
python gerar_video.py
```

Gera `AUD-20260506-WA0053_video.mp4` no estilo de editores de áudio (tipo Wavepad): a forma de onda completa em fundo azul-escuro, com a porção já tocada em ciano vivo e a porção futura em cinza esmaecido, separadas por um playhead branco que avança sincronizado ao som. 30 fps por padrão; ajustável com `--fps`. Renderização ~10 s para um áudio de 63 s.

---

## Parâmetros técnicos

| Parâmetro | Valor |
|---|---|
| Taxa de amostragem (`sr`) | 16.000 Hz |
| Coeficientes Mel (`n_mels`) | 128 |
| Frequência máxima (`fmax`) | 8.000 Hz |
| Colormap | `magma` |

---

## Como ouvir o áudio

A gravação está incluída em dois formatos:

- **`AUD-20260506-WA0053.wav`** — para tocar **direto no GitHub** (clique no arquivo na interface web e o player nativo aparece). 16 kHz mono, ~2 MB.
- **`AUD-20260506-WA0053.opus`** — formato original do WhatsApp, mais compacto (~130 KB), mas o GitHub **não** o toca inline. Use VLC, mpv, ffplay, ou abra com `librosa.load(...)` em Python.

Os dois arquivos contêm o mesmo conteúdo sonoro; o `.wav` é apenas uma re-codificação do `.opus` em PCM 16 bit. As imagens deste projeto foram geradas a partir do `.opus`, mas o resultado é idêntico se `gerar_imagens.py` for chamado com `--audio AUD-20260506-WA0053.wav`.

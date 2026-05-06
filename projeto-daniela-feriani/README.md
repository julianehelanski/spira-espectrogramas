# projeto-daniela-feriani

Subprojeto isolado dentro do repositório `spira-espectrogramas`. Reúne a gravação `AUD-20260506-WA0053.opus` e os três produtos visuais derivados dela, gerados com as mesmas funções e os mesmos parâmetros técnicos do SPIRA (sr=16.000 Hz, n_mels=128, fmax=8.000 Hz, cmap=`magma`).

---

## Arquivos

| Arquivo | Tipo | Descrição |
|---|---|---|
| `AUD-20260506-WA0053.opus` | áudio | Gravação original (63,14 s, formato `.opus` do WhatsApp) |
| `AUD-20260506-WA0053_waveform.png` | imagem | Forma de onda bruta — linha branca sobre fundo preto, sem eixos |
| `AUD-20260506-WA0053_mel_sem_legenda.png` | imagem | Espectrograma mel 128 × T em dB, sem eixos nem barra de cor |
| `AUD-20260506-WA0053_mel_com_eixos.png` | imagem | Espectrograma mel com Tempo (s), Frequência (Mel) e barra de dB |

---

## O que cada gráfico é e faz

### Forma de onda (`_waveform.png`)

Plot do sinal de áudio bruto: cada ponto horizontal é uma amostra de pressão do ar capturada pelo microfone (uma a cada 1/16.000 s); o eixo vertical é a amplitude instantânea normalizada. Mostra o polo mais material da cadeia, antes de qualquer decomposição em frequências. Permite ler a estrutura temporal da fala — blocos densos de oscilação são sílabas vocalizadas, zonas finas próximas ao zero são pausas, picos isolados são oclusivas ou ruídos transientes. Não revela conteúdo espectral.

### Espectrograma mel sem eixos (`_mel_sem_legenda.png`)

Imagem bidimensional 128 × T da matriz mel-espectral em decibéis. Eixo horizontal: tempo em frames. Eixo vertical: 128 bandas perceptuais Mel entre 0 e 8.000 Hz. Cor (`magma`): energia em dB. É a "imagem do som" como ela entra em uma CNN: textura espectral pura, antes da nomeação das coordenadas. Concentrações horizontais brilhantes nas bandas inferiores são formantes vocálicos; estrias verticais marcam consoantes e ataques transientes; faixas pretas verticais marcam pausas respiratórias.

### Espectrograma mel com eixos (`_mel_com_eixos.png`)

Mesma matriz da figura anterior, mas com eixos calibrados — Tempo (s) no horizontal, Frequência (Mel) no vertical — e barra de cor em decibéis (`+0 dB` no topo, valores negativos abaixo indicando atenuação relativa). É a forma canônica de circulação científica do espectrograma: as coordenadas nomeadas convertem a textura em objeto comparável entre laboratórios e citável em publicações.

---

## Reprodução

A partir da raiz do repositório (`spira-espectrogramas/`):

```bash
pip install -r requirements.txt

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

| Parâmetro | Valor |
|---|---|
| Taxa de amostragem (`sr`) | 16.000 Hz |
| Coeficientes Mel (`n_mels`) | 128 |
| Frequência máxima (`fmax`) | 8.000 Hz |
| Colormap | `magma` |

Idênticos aos do projeto SPIRA (ver README principal do repositório para referências).

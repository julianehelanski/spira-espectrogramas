# gerar_espectrogramas_spira

Script de geração das formas de onda e espectrogramas mel utilizados no Capítulo 4 da dissertação **"A rede que Marcelo construiu"** (Helanski, 2026), a partir de gravações do dataset público do projeto SPIRA (IME-USP / C4AI-USP).

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

Os pares sem eixos / com eixos reproduzem o gesto analítico de tornar visível a cadeia de transformação do sinal acústico em inscrição circulável, nos termos de Latour (2001).

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

Versões mínimas testadas: Python 3.10, librosa 0.10, matplotlib 3.7, numpy 1.24.

---

## Uso

```bash
python gerar_espectrogramas_spira.py \
    --controle caminho/para/controle.wav \
    --paciente caminho/para/PTT-20200511-WA0018.wav \
    --saida    figuras/cap.4/
```

Se os argumentos forem omitidos, o script busca os arquivos no diretório corrente com os nomes padrão `spira_controle.wav` e `PTT-20200511-WA0018.wav`, e salva as figuras no diretório corrente.

---

## Cadeia de transformação

O script expõe as três etapas da cadeia de transformação do sinal acústico em inscrição:

1. **Forma de onda** (`salvar_waveform`): o sinal bruto como sequência de amostras de pressão do ar. Polo mais material da cadeia, antes de qualquer decomposição espectral.
2. **Espectrograma mel sem eixos** (`salvar_sem_eixos`): a matriz bidimensional 128 × T visualizada como textura espectral, antes da nomeação das coordenadas.
3. **Espectrograma mel com eixos** (`salvar_com_eixos`): o mesmo objeto com tempo (s), frequência (Mel) e amplitude (dB) nomeados, no estado em que pode circular entre laboratórios.

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

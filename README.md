# Espectrogramas SPIRA — material suplementar

Script Python para geração dos quatro espectrogramas mel utilizados no Capítulo 4 da dissertação de doutorado **"A rede que Marcelo construiu"**, parte da tese *[título da tese]* (Helanski, 2026), Departamento de Antropologia, IFCH, Universidade Estadual de Campinas.

As figuras documentam a operação que a cadeia de referência circulante realiza sobre o sinal acústico do projeto SPIRA: vozes coletadas em enfermaria de COVID-19 e por aplicativo web são convertidas em texturas bidimensionais que uma rede neural convolucional aprende a classificar como insuficiência respiratória presente ou ausente.

---

## Figuras geradas

| Arquivo | Descrição |
|---|---|
| `spira_controle_sem_legenda.png` | Espectrograma mel — grupo controle, sem eixos |
| `spira_controle_com_eixos.png` | Espectrograma mel — grupo controle, com eixos |
| `spira_paciente_sem_legenda.png` | Espectrograma mel — grupo paciente (IR), sem eixos |
| `spira_paciente_com_eixos.png` | Espectrograma mel — grupo paciente (IR), com eixos |

Parâmetros técnicos: `sr=16000 Hz` | `n_mels=128` | `fmax=8000 Hz` | `cmap=magma`. Reproduzem o padrão descrito em Casanova Gris et al. (2021) e Gauy et al. (2024) — ver Referências.

---

## Dataset

As gravações de áudio utilizadas pertencem ao dataset público do projeto SPIRA (IME-USP / C4AI-USP), disponível sob licença **CC BY-SA 4.0**.

- Repositório principal (código-fonte e dataset, ACL 2021): [github.com/SPIRA-COVID19/SPIRA-ACL2021](https://github.com/SPIRA-COVID19/SPIRA-ACL2021)
- Organização geral do projeto: [github.com/spirabr](https://github.com/spirabr)
- Áudios de fala (pacientes e controles): [Google Drive](https://drive.google.com/file/d/1Bv0d3uwBB-52MBmtN2A_qNoaBIxUkN9y/view)
- Ruídos de enfermaria hospitalar: [Google Drive](https://drive.google.com/file/d/1zNwkye2FhV5LOVh3OfdqgPKzmYS7LeCM/view)

O dataset foi coletado de forma completamente anônima. Os arquivos `.wav` são identificados apenas por metadados neutros: sexo, faixa etária, saturação de oxigênio no sangue e uso de máscara. A anonimização foi condição de possibilidade da publicação e circulação dos dados como recurso aberto.

---

## Dependências

```bash
pip install -r requirements.txt
```

Ou individualmente:

```bash
pip install librosa matplotlib numpy
```

---

## Uso

```bash
# com os arquivos de áudio no diretório corrente
python gerar_espectrogramas_spira.py

# especificando caminhos e diretório de saída
python gerar_espectrogramas_spira.py \
    --controle dados/controle.wav \
    --paciente dados/PTT-20200511-WA0018.wav \
    --saida    figuras/cap.4/
```

---

## Referências

**Artigos do projeto SPIRA:**

- Casanova Gris, E. et al. Towards a COVID-19 respiratory insufficiency detection system based on speech. *Findings of ACL-IJCNLP 2021*, p. 617–628. Disponível em: [aclanthology.org/2021.findings-acl.55](https://aclanthology.org/2021.findings-acl.55).

- Gauy, M. V. et al. Discriminant analysis for respiratory insufficiency using deep learning models and transfer learning. *Intelligence-Based Medicine*, 2024. Disponível em: [arxiv.org/abs/2511.14939](https://arxiv.org/abs/2511.14939).

**Dissertação que utiliza este material:**

- Helanski, J. *[Título da tese]*. Tese (Doutorado em Antropologia) — IFCH, Universidade Estadual de Campinas, Campinas, 2026.

---

## Nota metodológica

Este script foi produzido com auxílio do modelo de linguagem Claude Sonnet 4.6 (Anthropic, 2025–2026) em sessão de trabalho conduzida em 12 de março de 2026. A interpretação analítica das imagens geradas é da pesquisadora.

---

## Licença

O código deste repositório está disponível sob licença [MIT](LICENSE). As gravações de áudio utilizadas pertencem ao dataset SPIRA, disponível sob licença [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

# Gravações de áudio

Esta pasta guarda as três gravações do *dataset* público do projeto SPIRA que geram as figuras do capítulo 4 da tese (formas de onda e espectrogramas mel). O *dataset* completo não está no repositório.

## Procedência

Caminho de acesso:

1. Repositório do projeto no GitHub: [`Spira-COVID19/SPIRA-ACL2021`](https://github.com/Spira-COVID19/SPIRA-ACL2021), seção *Datasets Download* do README.
2. Link dessa seção para o **Speech COVID-19 Dataset**, no Google Drive: <https://drive.google.com/file/d/1Bv0d3uwBB-52MBmtN2A_qNoaBIxUkN9y/view?usp=sharing>.
3. Do arquivo baixado foram retiradas as gravações abaixo, sem alteração.

O capítulo 4 da tese registra o acesso ao repositório em 11 mar. 2026 e ao Google Drive em 12 mar. 2026.

| Arquivo | Grupo | Duração | Uso na tese |
|---|---|---|---|
| `22e8506a-9916-49b9-ac5d-21b397276e4a_1.wav` | controle | 12,26 s | figuras do controle desde jul. 2026 (forma de onda e espectrogramas) |
| `0a2d6271-846b-4157-a784-b5fa2d93d2f9_1.wav` | controle | 8,53 s | figuras do controle até jul. 2026; mantida como registro da primeira versão |
| `PTT-20200511-WA0018.wav` | paciente | 10,48 s | figuras do paciente (forma de onda, espectrogramas, comparação linear × logarítmica) |

As durações foram medidas com `librosa.load(..., sr=16000)`. Os nomes dos arquivos são os do *dataset*:

- **controles:** identificador único (UUID) gerado pela aplicação web de coleta;
- **paciente:** nome atribuído pelo WhatsApp, com a data de transmissão (11 mai. 2020).

Nenhum dos nomes identifica a pessoa que gravou.

## Licença e atribuição

O *dataset* é distribuído sob a licença [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). As gravações desta pasta seguem a mesma licença. Ao reutilizá-las, cite:

> Casanova, E. et al. (2021). Deep learning against COVID-19: respiratory insufficiency detection in Brazilian Portuguese speech. *Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021*, p. 625–633.

## Uso

Rodado na raiz do repositório sem argumentos, `gerar_espectrogramas_spira.py` usa `audios/22e8506a-9916-49b9-ac5d-21b397276e4a_1.wav` como controle e `audios/PTT-20200511-WA0018.wav` como paciente. Para a versão anterior das figuras do controle:

```bash
python gerar_espectrogramas_spira.py --controle audios/0a2d6271-846b-4157-a784-b5fa2d93d2f9_1.wav
```

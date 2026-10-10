# Espectrogramas do Spira

Este repositório guarda o *script* que gera as formas de onda e os espectrogramas mel do capítulo 4, "A rede que Marcelo construiu", da minha tese de doutorado, *Tecnografias de um centro de inteligência artificial: seguindo cientistas e engenheiros universidade afora* (Programa de Pós-Graduação em Ciências Sociais, IFCH, Unicamp, 2026). As imagens são feitas a partir de gravações do *dataset* público do projeto SPIRA (IME-USP/C4AI-USP), o sistema de detecção de insuficiência respiratória pela voz que acompanho no capítulo.

## O que fiz

Escrevi `gerar_espectrogramas_spira.py` para refazer, sobre gravações do próprio *dataset* do SPIRA, a cadeia de transformação que o sistema opera sobre a voz: do sinal acústico à forma de onda, da forma de onda ao espectrograma mel e do espectrograma à varredura de uma rede neural convolucional. Usei os parâmetros descritos nos artigos do projeto (taxa de amostragem de 16.000 Hz, 128 bandas mel, frequência máxima de 8.000 Hz) e o mapa de cor `viridis`. O *script* produz cada espectrograma em dois estados, sem eixos e com eixos (tempo em segundos, frequência em mel, amplitude em dB), para tornar visível o momento em que a imagem do som ganha coordenadas e passa a circular como inscrição.

Escolhi duas gravações: uma do grupo controle (`22e8506a-9916-49b9-ac5d-21b397276e4a_1.wav`, 12,26 s) e uma do grupo paciente (`PTT-20200511-WA0018.wav`, 10,48 s).

## O que entra na tese

No capítulo 4, subseção "Espectrogramas mel: a imagem da voz", e na abertura da seção "Seguindo Marcelo", entram sete imagens (oito figuras, porque o espectrograma do paciente sem eixos aparece duas vezes), todas em `figuras/cap.4/`:

| Arquivo | Figura |
|---|---|
| `spira_waveform_controle.png` | forma de onda, grupo controle |
| `spira_waveform_paciente.png` | forma de onda, grupo paciente |
| `spira_controle_sem_legenda.png` | espectrograma mel, controle, sem eixos |
| `spira_controle_com_eixos.png` | espectrograma mel, controle, com eixos e barra de cor |
| `spira_paciente_sem_legenda.png` | espectrograma mel, paciente, sem eixos (também na abertura da seção "Seguindo Marcelo") |
| `spira_paciente_com_eixos.png` | espectrograma mel, paciente, com eixos e barra de cor |
| `spira_comparacao_linear_log.png` | comparação entre as escalas linear e logarítmica de frequência |

O *script* gera também `spira_cnn_diagrama.png` (espectrograma do paciente, filtro convolucional 3×3 e mapa de ativação), que não entra na tese. Os blocos de código reproduzidos no capítulo vêm deste *script*. A correspondência figura a figura está em [`docs/USO_NA_TESE.md`](docs/USO_NA_TESE.md) (versão tabular em [`docs/uso_na_tese.csv`](docs/uso_na_tese.csv)).

## Dados

As gravações não são distribuídas aqui. Pertencem ao *dataset* público do SPIRA, sob licença CC BY-SA 4.0, e a procedência, a duração e a forma de obtenção de cada uma estão em [`audios/README.md`](audios/README.md).

| Recurso | Endereço |
|---|---|
| Repositório do projeto (código e *dataset*, ACL 2021) | https://github.com/SPIRA-COVID19/SPIRA-ACL2021 |
| Áudios de fala (pacientes e controles) | https://drive.google.com/file/d/1Bv0d3uwBB-52MBmtN2A_qNoaBIxUkN9y/view |

## Como reproduzir

```bash
pip install -r requirements.txt      # Python 3.10+, librosa, matplotlib, numpy, scipy

# com as duas gravações em audios/, gera todas as figuras
python gerar_espectrogramas_spira.py --saida figuras/cap.4/

# com outras gravações
python gerar_espectrogramas_spira.py --controle caminho/controle.wav \
    --paciente caminho/paciente.wav --saida figuras/cap.4/

# apenas o diagrama CNN, com espectrograma simulado (sem .wav)
python gerar_espectrogramas_spira.py --apenas-cnn --saida figuras/cap.4/
```

## Uso de inteligência artificial generativa

Escrevi o *script* com o Claude e o revisei com o Claude Code. O Claude Code é a interface de linha de comando da Anthropic que dá ao modelo de linguagem acesso aos arquivos do projeto, para ler, escrever e executar *scripts*. São minhas a escolha das gravações e dos parâmetros e a interpretação das imagens no capítulo 4, onde descrevo também o percurso desse trabalho com o modelo de linguagem (subseção "Localização do *dataset* e geração dos espectrogramas").

**Modelos registrados no histórico de versões:** Claude Sonnet 4.6, Claude Opus 4.8, Claude Opus 5.5 e Claude Sonnet 5.5 (março a outubro de 2026).

Os *commits* com autor `Claude`, ou com a linha `Co-Authored-By: Claude …`, foram feitos em sessões do Claude Code; a marcação é gerada pela ferramenta e registra em que pontos do histórico o modelo participou do trabalho. A autoria e a responsabilidade pelo conteúdo são minhas e, conforme a Deliberação CONSU-A-005/2026 da Unicamp, as ferramentas de IA generativa não figuram como coautoras. A declaração formal de uso de IA generativa da tese está no [Anexo 1](https://github.com/julianehelanski/tecno-etnografia-centro-ia/blob/main/ex_ane1.tex).

## Citação

> CARDOSO, Juliane Cristina Helanski. *Tecnografias de um centro de inteligência artificial*: seguindo cientistas e engenheiros universidade afora. Orientadora: Maria Suely Kofes. 2026. Tese (Doutorado em Ciências Sociais) – Instituto de Filosofia e Ciências Humanas, Universidade Estadual de Campinas, Campinas, 2026.

> CASANOVA, Edresson *et al.* Deep learning against COVID-19: respiratory insufficiency detection in Brazilian Portuguese speech. In: *Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021*. [S. l.]: ACL, 2021. p. 625–633. Disponível em: https://aclanthology.org/2021.findings-acl.55.

ORCID da autora: https://orcid.org/0000-0001-8649-8986.

Metadados de citação em [`CITATION.cff`](CITATION.cff).

## Licença

Código sob licença MIT. As gravações do SPIRA seguem a licença CC BY-SA 4.0 do *dataset*.

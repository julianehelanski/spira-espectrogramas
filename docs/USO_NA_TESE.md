# Uso deste repositório na tese

Documento gerado em 09/10/2026 a partir da leitura dos arquivos `ex_cap*.tex` do repositório da tese (`julianehelanski/tecno-etnografia-centro-ia`, commit 3f0f671 (2026-10-08)). Repositório descrito: `julianehelanski/spira-espectrogramas`. A versão tabular está em `docs/uso_na_tese.csv`.

## Onde entra na tese

Capítulo 4, subseção "Espectrogramas mel: a imagem da voz" (`subsec:voz_espectrograma`), e a dedicatória-abertura da seção "Seguindo Marcelo". A tese cita este repositório em `ex_cap4.tex` (nota sobre os blocos de código produzidos com o claude code) com o endereço `Spira-espectrogramas`.

## Como os dados foram usados

Duas gravações do dataset público do projeto SPIRA (IME-USP/C4AI-USP; CC BY-SA 4.0), uma do grupo controle e uma do grupo paciente, são convertidas em forma de onda e em espectrograma mel (`sr` 16.000 Hz, 128 bandas mel, `fmax` 8.000 Hz, mapa de cor `viridis`, forma de onda em gradiente por amplitude sobre fundo branco) por `gerar_espectrogramas_spira.py`. Os pares sem eixos e com eixos reproduzem na tese o gesto de tornar visível a cadeia de transformação do sinal acústico em inscrição circulável. Os blocos de código reproduzidos na seção foram escritos com o claude code e aparecem como listagens no capítulo 4.

## Figuras da tese que vêm deste repositório

| Capítulo | Seção da tese | Rótulo | Arquivo no repositório | Script | Estado da cópia na tese |
|---|---|---|---|---|---|
| capítulo 4 | Seguindo Marcelo | `fig:espectrograma-paciente-dedicatoria` | `figuras/cap.4/spira_paciente_sem_legenda.png` | `gerar_espectrogramas_spira.py` | cópia na tese idêntica à do repositório |
| capítulo 4 | Espectogramas mel: a imagem da voz | `fig:waveform-controle` | `figuras/cap.4/spira_waveform_controle.png` | `gerar_espectrogramas_spira.py` | cópia na tese idêntica à do repositório |
| capítulo 4 | Espectogramas mel: a imagem da voz | `fig:espectrograma-sem-legenda` | `figuras/cap.4/spira_controle_sem_legenda.png` | `gerar_espectrogramas_spira.py` | cópia na tese idêntica à do repositório |
| capítulo 4 | Espectogramas mel: a imagem da voz | `fig:espectrograma-com-eixos` | `figuras/cap.4/spira_controle_com_eixos.png` | `gerar_espectrogramas_spira.py` | cópia na tese idêntica à do repositório |
| capítulo 4 | Espectogramas mel: a imagem da voz | `fig:waveform-paciente` | `figuras/cap.4/spira_waveform_paciente.png` | `gerar_espectrogramas_spira.py` | cópia na tese idêntica à do repositório |
| capítulo 4 | Espectogramas mel: a imagem da voz | `fig:paciente-com-eixos` | `figuras/cap.4/spira_paciente_com_eixos.png` | `gerar_espectrogramas_spira.py` | cópia na tese idêntica à do repositório |
| capítulo 4 | Espectogramas mel: a imagem da voz | `fig:paciente-sem-legenda` | `figuras/cap.4/spira_paciente_sem_legenda.png` | `gerar_espectrogramas_spira.py` | cópia na tese idêntica à do repositório |
| capítulo 4 | Espectogramas mel: a imagem da voz | `fig:comparacao-linear-log` | `figuras/cap.4/spira_comparacao_linear_log.png` | `gerar_espectrogramas_spira.py` | cópia na tese idêntica à do repositório |

Nota de sincronização. Em julho de 2026 as figuras do capítulo 4 passaram da paleta magma para viridis, com a forma de onda em gradiente por amplitude sobre fundo branco, e o grupo controle passou a usar a gravação `22e8506a-…_1.wav` (12,26 s) do dataset público. Essa mudança foi feita só nas imagens da tese. Em 09/10/2026 alinhei o repositório a ela: o script passou a viridis com o gradiente das listagens do capítulo, e as figuras citadas na tese foram copiadas da tese para `figuras/cap.4/` (os espectrogramas do controle com os nomes `spira_controle_*`), de modo que as cópias são idênticas. Regenerado com as mesmas gravações, o script reproduz a forma de onda pixel a pixel; os espectrogramas saem com o mesmo conteúdo e dimensões em pixels um pouco maiores. A gravação de controle `22e8506a-…_1.wav` ainda não está no repositório: a da raiz (`0a2d6271-…_1.wav`, 8,53 s) é a usada até julho de 2026.

## Dados e consentimento

O README do repositório afirma que o dataset não está incluído, mas a raiz contém dois arquivos de áudio do dataset público: `0a2d6271-846b-4157-a784-b5fa2d93d2f9_1.wav` (controle, identificador gerado pela aplicação web) e `PTT-20200511-WA0018.wav` (paciente, nome do WhatsApp com data da transmissão). A subpasta `projeto-daniela-feriani/` contém uma gravação de voz individual (WhatsApp, 06/05/2026) com os derivados em imagem e vídeo; ela não faz parte do dataset SPIRA e não aparece no capítulo 4. Confirmar a base de consentimento e de licença desses arquivos antes do depósito.

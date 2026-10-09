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

Nota de sincronização. Em julho de 2026 as figuras do capítulo 4 passaram da paleta magma para viridis, com a forma de onda em gradiente por amplitude sobre fundo branco, e o grupo controle passou a usar a gravação `22e8506a-…_1.wav` (12,26 s) do dataset público. Essa mudança foi feita só nas imagens da tese. Em 09/10/2026 alinhei o repositório a ela: o script passou a viridis com o gradiente das listagens do capítulo, e as figuras citadas na tese foram copiadas da tese para `figuras/cap.4/` (os espectrogramas do controle com os nomes `spira_controle_*`), de modo que as cópias são idênticas. Regenerado com as mesmas gravações, o script reproduz a forma de onda pixel a pixel; os espectrogramas saem com o mesmo conteúdo e dimensões em pixels um pouco maiores. A gravação de controle `22e8506a-9916-49b9-ac5d-21b397276e4a_1.wav` (12,26 s) está em `audios/` desde 09/10/2026; regenerada a partir dela, a forma de onda do controle sai pixel a pixel igual à da tese, e os espectrogramas do controle têm o mesmo conteúdo. A gravação `0a2d6271-846b-4157-a784-b5fa2d93d2f9_1.wav` (8,53 s) é a de controle usada até julho de 2026 e permanece como registro dessa primeira versão.

## Dados e consentimento

A pasta `audios/` contém as três gravações do dataset público SPIRA usadas nas figuras (CC BY-SA 4.0): `22e8506a-9916-49b9-ac5d-21b397276e4a_1.wav` (controle usado na tese desde julho de 2026), `0a2d6271-846b-4157-a784-b5fa2d93d2f9_1.wav` (controle usado até julho de 2026) e `PTT-20200511-WA0018.wav` (paciente). Os controles têm identificador gerado pela aplicação web de coleta; o paciente, o nome do WhatsApp com a data da transmissão. A subpasta `projeto-daniela-feriani/`, com uma gravação individual que não fazia parte do dataset nem da tese, foi retirada do repositório pela autora em 09/10/2026.

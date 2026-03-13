"""
gerar_espectrogramas_spira.py
─────────────────────────────────────────────────────────────────────────────
Geração dos quatro espectrogramas mel utilizados no Capítulo 4 da dissertação
"A rede que Marcelo construiu" (Helanski, 2026), a partir de gravações do
dataset público do projeto SPIRA (IME-USP / C4AI-USP).

As figuras geradas correspondem às Figuras apresentadas na seção
"A imagem do som: espectrogramas como inscrições" do Capítulo 4:

    Fig. 1 — spira_waveform_controle.png         (grupo controle, forma de onda)
    Fig. 2 — spira_waveform_paciente.png          (grupo paciente, forma de onda)
    Fig. 3 — spira_controle_sem_legenda.png       (grupo controle, sem eixos)
    Fig. 4 — spira_controle_com_eixos.png         (grupo controle, com eixos)
    Fig. 5 — spira_paciente_sem_legenda.png        (grupo paciente, sem eixos)
    Fig. 6 — spira_paciente_com_eixos.png          (grupo paciente, com eixos)

Os parâmetros técnicos (sr=16000, n_mels=128, fmax=8000) reproduzem o padrão
descrito nos artigos do projeto:

    Casanova Gris et al. (2021). Towards a COVID-19 respiratory insufficiency
    detection system based on speech. Findings of ACL-IJCNLP 2021, p. 617–628.
    https://aclanthology.org/2021.findings-acl.55

    Gauy et al. (2024). Discriminant analysis for respiratory insufficiency
    using deep learning models and transfer learning.
    arXiv:2511.14939. https://arxiv.org/abs/2511.14939

─────────────────────────────────────────────────────────────────────────────
Dataset público SPIRA
─────────────────────────────────────────────────────────────────────────────
Repositório principal (código-fonte e dataset, ACL 2021):
    https://github.com/SPIRA-COVID19/SPIRA-ACL2021
    Licença: CC BY-SA 4.0

Organização geral do projeto:
    https://github.com/spirabr

Áudios de fala (pacientes e controles) via Google Drive:
    https://drive.google.com/file/d/1Bv0d3uwBB-52MBmtN2A_qNoaBIxUkN9y/view

Ruídos de enfermaria hospitalar via Google Drive:
    https://drive.google.com/file/d/1zNwkye2FhV5LOVh3OfdqgPKzmYS7LeCM/view

─────────────────────────────────────────────────────────────────────────────
Dependências
─────────────────────────────────────────────────────────────────────────────
    pip install librosa matplotlib numpy

─────────────────────────────────────────────────────────────────────────────
Uso
─────────────────────────────────────────────────────────────────────────────
    python gerar_espectrogramas_spira.py \\
        --controle caminho/para/controle.wav \\
        --paciente caminho/para/PTT-20200511-WA0018.wav \\
        --saida    figuras/cap.4/

    Se os argumentos forem omitidos, o script busca os arquivos no diretório
    corrente com os nomes padrão informados abaixo.

─────────────────────────────────────────────────────────────────────────────
Nota metodológica
─────────────────────────────────────────────────────────────────────────────
Este script foi produzido com auxílio do modelo de linguagem Claude Sonnet 4.6
(Anthropic, 2025–2026) em sessão de trabalho conduzida em 12 de março de 2026.
A interpretação analítica das imagens geradas é da pesquisadora.
"""

import argparse
import os
import sys

import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Fonte sem serifa para compatibilidade com LaTeX (sans-serif)
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]

# ─────────────────────────────────────────────────────────────────────────────
# Parâmetros técnicos SPIRA
# ─────────────────────────────────────────────────────────────────────────────
SR = 16_000     # taxa de amostragem (Hz)
N_MELS = 128    # número de coeficientes Mel
FMAX = 8_000    # frequência máxima (Hz)
CMAP = "magma"  # colormap (reproduz o padrão das figuras da dissertação)

# Nomes de arquivo padrão
CONTROLE_DEFAULT = "spira_controle.wav"
PACIENTE_DEFAULT = "PTT-20200511-WA0018.wav"
SAIDA_DEFAULT    = "."


# ─────────────────────────────────────────────────────────────────────────────
# Funções auxiliares
# ─────────────────────────────────────────────────────────────────────────────

def carregar_audio(caminho: str) -> tuple[np.ndarray, int]:
    """Carrega arquivo .wav com sample rate padronizado para 16 kHz."""
    if not os.path.isfile(caminho):
        sys.exit(
            f"[ERRO] Arquivo não encontrado: {caminho}\n"
            "Informe o caminho correto via --controle ou --paciente."
        )
    y, sr = librosa.load(caminho, sr=SR)
    duracao = len(y) / sr
    print(f"  Arquivo: {caminho}")
    print(f"  Duração: {duracao:.2f} s  |  Sample rate: {sr} Hz")
    return y, sr


def salvar_waveform(
    y: np.ndarray,
    sr: int,
    caminho_saida: str,
) -> None:
    """
    Gera a forma de onda (waveform) do sinal bruto sem eixos.

    Representa o polo mais material da cadeia de referência circulante:
    o sinal antes de qualquer transformação espectral. Cada valor em y
    é uma amostra de pressão do ar em 1/16000 de segundo. Fundo preto
    para consistência visual com os espectrogramas sem eixos.
    """
    times = np.linspace(0, len(y) / sr, num=len(y))
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(times, y, color="white", linewidth=0.3, alpha=0.85)
    ax.set_axis_off()
    # limites verticais levemente expandidos para evitar corte
    margin = np.max(np.abs(y)) * 0.1
    ax.set_ylim(-np.max(np.abs(y)) - margin, np.max(np.abs(y)) + margin)
    plt.tight_layout(pad=0)
    plt.savefig(
        caminho_saida,
        dpi=150,
        bbox_inches="tight",
        pad_inches=0,
        facecolor="black",
    )
    plt.close()
    print(f"  Salvo: {caminho_saida}")


def calcular_espectrograma(y: np.ndarray, sr: int) -> np.ndarray:
    """Calcula o espectrograma mel em dB com os parâmetros SPIRA."""
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=FMAX)
    return librosa.power_to_db(S, ref=np.max)


def salvar_sem_eixos(
    S_dB: np.ndarray,
    sr: int,
    caminho_saida: str,
) -> None:
    """
    Gera espectrograma sem eixos nem legenda.

    Reproduz o modo como o objeto se apresentou antes da nomeação analítica:
    uma textura espectral sem palavras. Fundo preto para uso em apresentação.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    librosa.display.specshow(
        S_dB, sr=sr, x_axis=None, y_axis=None, cmap=CMAP, ax=ax
    )
    ax.set_axis_off()
    plt.tight_layout(pad=0)
    plt.savefig(
        caminho_saida,
        dpi=150,
        bbox_inches="tight",
        pad_inches=0,
        facecolor="black",
    )
    plt.close()
    print(f"  Salvo: {caminho_saida}")


def salvar_com_eixos(
    S_dB: np.ndarray,
    sr: int,
    caminho_saida: str,
    titulo: str = "",
) -> None:
    """
    Gera espectrograma com eixos nomeados e barra de cores.

    Torna visível a operação que a cadeia de referência circulante realiza
    sobre o sinal acústico: o que era voz torna-se textura bidimensional
    em espaço de coordenadas padronizadas.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    img = librosa.display.specshow(
        S_dB,
        sr=sr,
        x_axis="time",
        y_axis="mel",
        fmax=FMAX,
        cmap=CMAP,
        ax=ax,
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_xlabel("Tempo (s)", fontsize=12)
    ax.set_ylabel("Frequência (Mel)", fontsize=12)
    if titulo:
        ax.set_title(titulo, fontsize=12, pad=10)
    plt.tight_layout()
    plt.savefig(
        caminho_saida,
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()
    print(f"  Salvo: {caminho_saida}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Gera os seis espectrogramas e formas de onda do Capítulo 4 da dissertação "
            "'A rede que Marcelo construiu' (Helanski, 2026), "
            "a partir de gravações do dataset público SPIRA."
        )
    )
    parser.add_argument(
        "--controle",
        default=CONTROLE_DEFAULT,
        help=f"Caminho para o arquivo .wav do grupo controle "
             f"(padrão: {CONTROLE_DEFAULT})",
    )
    parser.add_argument(
        "--paciente",
        default=PACIENTE_DEFAULT,
        help=f"Caminho para o arquivo .wav do grupo paciente "
             f"(padrão: {PACIENTE_DEFAULT})",
    )
    parser.add_argument(
        "--saida",
        default=SAIDA_DEFAULT,
        help=f"Diretório de saída para as figuras (padrão: {SAIDA_DEFAULT})",
    )
    args = parser.parse_args()

    os.makedirs(args.saida, exist_ok=True)

    def caminho(nome: str) -> str:
        return os.path.join(args.saida, nome)

    # ── Grupo controle ────────────────────────────────────────────────────────
    print("\n[1/2] Grupo controle")
    y_c, sr_c = carregar_audio(args.controle)

    salvar_waveform(
        y_c,
        sr_c,
        caminho("spira_waveform_controle.png"),
    )

    S_c = calcular_espectrograma(y_c, sr_c)

    salvar_sem_eixos(
        S_c,
        sr_c,
        caminho("spira_controle_sem_legenda.png"),
    )
    salvar_com_eixos(
        S_c,
        sr_c,
        caminho("spira_controle_com_eixos.png"),
        titulo="Espectrograma Mel — grupo controle | 128 coeficientes, 16 kHz",
    )

    # ── Grupo paciente ────────────────────────────────────────────────────────
    print("\n[2/2] Grupo paciente")
    y_p, sr_p = carregar_audio(args.paciente)

    salvar_waveform(
        y_p,
        sr_p,
        caminho("spira_waveform_paciente.png"),
    )

    S_p = calcular_espectrograma(y_p, sr_p)

    salvar_sem_eixos(
        S_p,
        sr_p,
        caminho("spira_paciente_sem_legenda.png"),
    )
    salvar_com_eixos(
        S_p,
        sr_p,
        caminho("spira_paciente_com_eixos.png"),
        titulo="Espectrograma Mel — grupo paciente (IR) | 128 coeficientes, 16 kHz",
    )

    print(
        f"\nConcluído. Seis figuras salvas em: {os.path.abspath(args.saida)}"
    )
    print(
        "Parâmetros: sr=16000 Hz | n_mels=128 | fmax=8000 Hz | cmap=magma"
    )


if __name__ == "__main__":
    main()

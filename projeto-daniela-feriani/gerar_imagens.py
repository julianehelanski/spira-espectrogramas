"""
gerar_imagens.py
─────────────────────────────────────────────────────────────────────────────
Gera as três imagens do projeto a partir do arquivo de áudio
AUD-20260506-WA0053.opus:

    1. AUD-20260506-WA0053_waveform.png         — forma de onda
    2. AUD-20260506-WA0053_mel_sem_legenda.png  — espectrograma mel sem eixos
    3. AUD-20260506-WA0053_mel_com_eixos.png    — espectrograma mel com eixos e dB

Parâmetros técnicos: sr=16000 Hz, n_mels=128, fmax=8000 Hz, colormap=magma.

─────────────────────────────────────────────────────────────────────────────
Dependências
─────────────────────────────────────────────────────────────────────────────
    pip install librosa matplotlib numpy

    Para decodificar arquivos .opus, librosa precisa de um backend de áudio
    como ffmpeg (já incluído na maioria dos sistemas) ou soundfile.

─────────────────────────────────────────────────────────────────────────────
Uso
─────────────────────────────────────────────────────────────────────────────
    # roda com os caminhos padrão (áudio na mesma pasta do script)
    python gerar_imagens.py

    # caminho de áudio customizado
    python gerar_imagens.py --audio outro_arquivo.opus --saida ./

─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import os
import sys

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Parâmetros técnicos
# ─────────────────────────────────────────────────────────────────────────────
SR = 16_000     # taxa de amostragem (Hz)
N_MELS = 128    # número de coeficientes Mel
FMAX = 8_000    # frequência máxima (Hz)
CMAP = "magma"  # colormap

AUDIO_DEFAULT = "AUD-20260506-WA0053.opus"
SAIDA_DEFAULT = "."


def carregar_audio(caminho: str) -> tuple[np.ndarray, int]:
    """Carrega um arquivo de áudio e o reamostra para 16 kHz mono."""
    if not os.path.isfile(caminho):
        sys.exit(f"[ERRO] Arquivo não encontrado: {caminho}")
    y, sr = librosa.load(caminho, sr=SR)
    duracao = len(y) / sr
    print(f"  Arquivo : {caminho}")
    print(f"  Duração : {duracao:.2f} s  |  Sample rate: {sr} Hz")
    return y, sr


def salvar_waveform(y: np.ndarray, sr: int, caminho_saida: str) -> None:
    """
    Plota a forma de onda do sinal bruto: amostras de pressão do ar ao
    longo do tempo. Linha branca sobre fundo preto, sem eixos.
    """
    times = np.linspace(0, len(y) / sr, num=len(y))
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(times, y, color="white", linewidth=0.3, alpha=0.85)
    ax.set_axis_off()
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
    """Calcula o espectrograma mel em decibéis (dB)."""
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=FMAX)
    return librosa.power_to_db(S, ref=np.max)


def salvar_mel_sem_eixos(S_dB: np.ndarray, sr: int, caminho_saida: str) -> None:
    """Plota o espectrograma mel sem eixos nem barra de cor (textura pura)."""
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


def salvar_mel_com_eixos(
    S_dB: np.ndarray,
    sr: int,
    caminho_saida: str,
    titulo: str = "",
) -> None:
    """Plota o espectrograma mel com eixos Tempo (s) × Frequência (Mel) e barra de dB."""
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera forma de onda e espectrogramas mel a partir de um áudio."
    )
    parser.add_argument(
        "--audio",
        default=AUDIO_DEFAULT,
        help=f"Caminho para o arquivo de áudio (padrão: {AUDIO_DEFAULT})",
    )
    parser.add_argument(
        "--saida",
        default=SAIDA_DEFAULT,
        help=f"Diretório de saída para os PNGs (padrão: {SAIDA_DEFAULT})",
    )
    args = parser.parse_args()

    os.makedirs(args.saida, exist_ok=True)

    base = os.path.splitext(os.path.basename(args.audio))[0]

    def caminho(sufixo: str) -> str:
        return os.path.join(args.saida, f"{base}{sufixo}")

    print("\n[1/3] Carregando áudio")
    y, sr = carregar_audio(args.audio)

    print("\n[2/3] Forma de onda")
    salvar_waveform(y, sr, caminho("_waveform.png"))

    print("\n[3/3] Espectrogramas mel")
    S = calcular_espectrograma(y, sr)
    salvar_mel_sem_eixos(S, sr, caminho("_mel_sem_legenda.png"))
    salvar_mel_com_eixos(
        S, sr,
        caminho("_mel_com_eixos.png"),
        titulo=f"Espectrograma Mel — {base} | 128 coeficientes, 16 kHz",
    )

    print(f"\nConcluído. Imagens salvas em: {os.path.abspath(args.saida)}")
    print(f"Parâmetros: sr={SR} Hz | n_mels={N_MELS} | fmax={FMAX} Hz | cmap={CMAP}")


if __name__ == "__main__":
    main()

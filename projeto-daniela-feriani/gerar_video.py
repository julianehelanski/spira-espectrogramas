"""
gerar_video.py
─────────────────────────────────────────────────────────────────────────────
Gera um vídeo MP4 da forma de onda do áudio com um indicador (linha vertical)
que se move sincronizado com a reprodução do som.

    entrada : AUD-20260506-WA0053.opus  (ou .wav)
    saída   : AUD-20260506-WA0053_video.mp4

Estratégia: a forma de onda é renderizada como imagem estática (PNG) uma
única vez. Em seguida, o ffmpeg compõe o vídeo aplicando um filtro
`drawbox` cuja posição horizontal varia com o tempo, e mixa o áudio
original na mesma passada. Isso é dezenas de vezes mais rápido do que
renderizar 1 frame por instante com matplotlib.

─────────────────────────────────────────────────────────────────────────────
Dependências
─────────────────────────────────────────────────────────────────────────────
    pip install librosa matplotlib numpy
    apt install ffmpeg

─────────────────────────────────────────────────────────────────────────────
Uso
─────────────────────────────────────────────────────────────────────────────
    python gerar_video.py
    python gerar_video.py --audio outro.opus --fps 30 --saida ./
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import librosa
import matplotlib.pyplot as plt
import numpy as np

SR = 16_000
AUDIO_DEFAULT = "AUD-20260506-WA0053.opus"
SAIDA_DEFAULT = "."
FPS_DEFAULT = 30
LARGURA = 1280
ALTURA = 320


def carregar_audio(caminho: str) -> tuple[np.ndarray, int]:
    if not os.path.isfile(caminho):
        sys.exit(f"[ERRO] Arquivo não encontrado: {caminho}")
    y, sr = librosa.load(caminho, sr=SR, mono=True)
    print(f"  Arquivo : {caminho}")
    print(f"  Duração : {len(y)/sr:.2f} s  |  Sample rate: {sr} Hz")
    return y, sr


def renderizar_waveform_png(
    y: np.ndarray,
    sr: int,
    caminho_png: str,
    largura: int,
    altura: int,
) -> None:
    """Salva a forma de onda como PNG estático com dimensões exatas em pixels."""
    duracao = len(y) / sr
    times = np.linspace(0, duracao, num=len(y))

    dpi = 100
    fig, ax = plt.subplots(figsize=(largura / dpi, altura / dpi), dpi=dpi)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.plot(times, y, color="white", linewidth=0.4, alpha=0.9)
    margin = np.max(np.abs(y)) * 0.1
    ax.set_xlim(0, duracao)
    ax.set_ylim(-np.max(np.abs(y)) - margin, np.max(np.abs(y)) + margin)
    ax.set_axis_off()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(caminho_png, dpi=dpi, facecolor="black", pad_inches=0)
    plt.close(fig)


def gerar_video(
    audio_path: str,
    saida_path: str,
    fps: int,
    largura: int,
    altura: int,
) -> None:
    if not shutil.which("ffmpeg"):
        sys.exit("[ERRO] ffmpeg não encontrado no PATH. Instale com 'apt install ffmpeg'.")

    y, sr = carregar_audio(audio_path)
    duracao = len(y) / sr

    with tempfile.TemporaryDirectory() as tmp:
        png = os.path.join(tmp, "waveform.png")
        print(f"\n[1/2] Renderizando imagem da forma de onda ({largura}x{altura})")
        renderizar_waveform_png(y, sr, png, largura, altura)

        # Barra vermelha de 3 px de largura como PNG separado para overlay
        playhead_png = os.path.join(tmp, "playhead.png")
        ph = np.zeros((altura, 3, 4), dtype=np.uint8)
        ph[:, :, 0] = 0xE7  # R
        ph[:, :, 1] = 0x4C  # G
        ph[:, :, 2] = 0x3C  # B
        ph[:, :, 3] = 230   # A (alpha)
        plt.imsave(playhead_png, ph)

        print(f"\n[2/2] Compondo vídeo com overlay do playhead e áudio (ffmpeg)")
        # `overlay` reconhece `t` (timestamp) em expressões — diferente de
        # `drawbox`, em que `t` colide com a opção `thickness`.
        x_expr = f"(t/{duracao})*({largura}-3)"
        filter_complex = f"[0:v][1:v]overlay=x={x_expr}:y=0[vout]"

        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-framerate", str(fps), "-t", f"{duracao}",
            "-i", png,
            "-loop", "1", "-framerate", str(fps), "-t", f"{duracao}",
            "-i", playhead_png,
            "-i", audio_path,
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-map", "2:a",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            saida_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            sys.exit(f"[ERRO] ffmpeg falhou:\n{result.stderr}")

    print(f"\nConcluído: {saida_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera um MP4 da forma de onda com áudio sincronizado."
    )
    parser.add_argument("--audio", default=AUDIO_DEFAULT)
    parser.add_argument("--saida", default=SAIDA_DEFAULT)
    parser.add_argument("--fps", type=int, default=FPS_DEFAULT)
    parser.add_argument("--largura", type=int, default=LARGURA)
    parser.add_argument("--altura", type=int, default=ALTURA)
    args = parser.parse_args()

    os.makedirs(args.saida, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.audio))[0]
    saida_path = os.path.join(args.saida, f"{base}_video.mp4")

    gerar_video(args.audio, saida_path, args.fps, args.largura, args.altura)


if __name__ == "__main__":
    main()

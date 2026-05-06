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
    cor: str = "white",
    alpha: float = 0.9,
    facecolor: str = "black",
) -> None:
    """Salva a forma de onda como PNG estático com dimensões exatas em pixels."""
    duracao = len(y) / sr
    times = np.linspace(0, duracao, num=len(y))

    dpi = 100
    fig, ax = plt.subplots(figsize=(largura / dpi, altura / dpi), dpi=dpi)
    fig.patch.set_facecolor(facecolor)
    ax.set_facecolor(facecolor)
    ax.plot(times, y, color=cor, linewidth=0.5, alpha=alpha)
    margin = np.max(np.abs(y)) * 0.1
    ax.set_xlim(0, duracao)
    ax.set_ylim(-np.max(np.abs(y)) - margin, np.max(np.abs(y)) + margin)
    ax.set_axis_off()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(caminho_png, dpi=dpi, facecolor=facecolor, pad_inches=0)
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
        # Renderiza duas versões da forma de onda:
        #   - "dim"  : a parte ainda não tocada (esmaecida)
        #   - "lit"  : a parte já tocada (cor viva)
        # Cada frame do vídeo combina pixels das duas conforme o tempo:
        # à esquerda do playhead, lit; à direita, dim.
        png_dim = os.path.join(tmp, "waveform_dim.png")
        png_lit = os.path.join(tmp, "waveform_lit.png")

        print(f"\n[1/3] Renderizando duas imagens da forma de onda ({largura}x{altura})")
        renderizar_waveform_png(
            y, sr, png_dim, largura, altura,
            cor="#5d6a7a", alpha=0.6, facecolor="#0e1116",
        )
        renderizar_waveform_png(
            y, sr, png_lit, largura, altura,
            cor="#3aaed8", alpha=1.0, facecolor="#0e1116",
        )

        dim = plt.imread(png_dim)
        lit = plt.imread(png_lit)
        # garantir uint8 RGB (matplotlib pode retornar float em [0,1] ou RGBA)
        def para_rgb_uint8(arr: np.ndarray) -> np.ndarray:
            if arr.dtype != np.uint8:
                arr = (arr * 255).astype(np.uint8)
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
            return np.ascontiguousarray(arr)
        dim = para_rgb_uint8(dim)
        lit = para_rgb_uint8(lit)
        H, W, _ = dim.shape

        n_frames = int(round(duracao * fps))
        print(f"\n[2/3] Renderizando {n_frames} frames (numpy)")

        cor_playhead = np.array([255, 255, 255], dtype=np.uint8)

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{W}x{H}", "-framerate", str(fps),
            "-i", "pipe:",
            "-i", audio_path,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            saida_path,
        ]

        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )

        for i in range(n_frames):
            x_t = int(round((i / n_frames) * W))
            frame = dim.copy()
            if x_t > 0:
                frame[:, :x_t] = lit[:, :x_t]
            # playhead 2 px branco
            x0 = max(0, x_t - 1)
            x1 = min(W, x_t + 1)
            frame[:, x0:x1] = cor_playhead
            proc.stdin.write(frame.tobytes())

        proc.stdin.close()
        rc = proc.wait()
        if rc != 0:
            err = proc.stderr.read().decode()
            sys.exit(f"[ERRO] ffmpeg falhou:\n{err}")
        print(f"\n[3/3] Vídeo finalizado")

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

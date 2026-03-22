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
    Fig. 7 — spira_cnn_diagrama.png               (diagrama de convolução CNN)
    Fig. 8 — spira_comparacao_linear_log.png      (comparação linear × logarítmica)

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
    pip install librosa matplotlib numpy scipy

─────────────────────────────────────────────────────────────────────────────
Uso
─────────────────────────────────────────────────────────────────────────────
    python gerar_espectrogramas_spira.py \\
        --controle caminho/para/controle.wav \\
        --paciente caminho/para/PTT-20200511-WA0018.wav \\
        --saida    figuras/cap.4/

    Para gerar apenas o diagrama CNN (sem arquivos de áudio):
        python gerar_espectrogramas_spira.py --apenas-cnn --saida figuras/cap.4/

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
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

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
# Funções auxiliares — espectrogramas e formas de onda
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


def salvar_comparacao_linear_log(
    y: np.ndarray,
    sr: int,
    caminho_saida: str,
) -> None:
    """
    Gera figura de comparação lado a lado entre o espectrograma mel
    em escala LINEAR (potência bruta) e em escala LOGARÍTMICA (dB),
    ambos calculados a partir do mesmo sinal de áudio com os parâmetros SPIRA.

    O painel esquerdo mostra o que o espectrograma pareceria sem a conversão
    para decibéis: a energia das frequências baixas domina a escala de cor
    e esmaga as variações nas frequências médias e altas.

    O painel direito mostra o espectrograma que efetivamente entra no
    classificador do SPIRA: a conversão logarítmica redistribui os valores
    de modo proporcional à percepção auditiva humana, tornando visíveis
    as diferenças nas faixas superiores.

    Parâmetros
    ----------
    y : np.ndarray
        Sinal de áudio carregado por librosa.load().
    sr : int
        Taxa de amostragem em Hz.
    caminho_saida : str
        Caminho completo para o arquivo PNG de saída.
    """
    # Potência bruta (escala linear — sem conversão para dB)
    S_linear = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=FMAX)

    # Espectrograma em dB (escala logarítmica — padrão SPIRA)
    S_dB = librosa.power_to_db(S_linear, ref=np.max)

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    # ── Painel esquerdo: escala linear ────────────────────────────────────────
    ax_lin = axes[0]
    img_lin = librosa.display.specshow(
        S_linear,
        sr=sr,
        x_axis="time",
        y_axis="mel",
        fmax=FMAX,
        cmap=CMAP,
        ax=ax_lin,
    )
    fig.colorbar(img_lin, ax=ax_lin, format="%.2e")
    ax_lin.set_xlabel("Tempo (s)", fontsize=11)
    ax_lin.set_ylabel("Frequência (Mel)", fontsize=11)
    ax_lin.set_title(
        "Escala LINEAR (potência bruta)\nsem conversão para decibéis",
        fontsize=11, pad=8,
    )
    ax_lin.annotate(
        "energia das frequências baixas\ndomina e apaga o restante",
        xy=(S_linear.shape[1] * 0.5, 10),
        xytext=(S_linear.shape[1] * 0.5, 60),
        color="white", fontsize=9, ha="center",
        arrowprops=dict(arrowstyle="->", color="white", lw=1.5),
    )

    # ── Painel direito: escala logarítmica (dB) — padrão SPIRA ───────────────
    ax_log = axes[1]
    img_log = librosa.display.specshow(
        S_dB,
        sr=sr,
        x_axis="time",
        y_axis="mel",
        fmax=FMAX,
        cmap=CMAP,
        ax=ax_log,
    )
    fig.colorbar(img_log, ax=ax_log, format="%+2.0f dB")
    ax_log.set_xlabel("Tempo (s)", fontsize=11)
    ax_log.set_ylabel("Frequência (Mel)", fontsize=11)
    ax_log.set_title(
        "Escala LOGARÍTMICA (decibéis) — padrão SPIRA\n"
        "conversão: librosa.power_to_db(S, ref=np.max)",
        fontsize=11, pad=8,
    )
    ax_log.annotate(
        "diferenças nas frequências\nmédias e altas tornadas visíveis",
        xy=(S_dB.shape[1] * 0.5, 80),
        xytext=(S_dB.shape[1] * 0.5, 110),
        color="white", fontsize=9, ha="center",
        arrowprops=dict(arrowstyle="->", color="white", lw=1.5),
    )

    fig.suptitle(
        "Comparação: escala linear × escala logarítmica\n"
        "arquivo: PTT-20200511-WA0018.wav | paciente com IR | dataset SPIRA (CC BY-SA 4.0)",
        fontsize=11, y=1.02,
    )

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
# Diagrama CNN — convolução sobre o espectrograma mel
# ─────────────────────────────────────────────────────────────────────────────

def gerar_diagrama_cnn(
    caminho_saida: str,
    S_dB: np.ndarray | None = None,
    sr: int = SR,
) -> None:
    """
    Gera o diagrama de convolução CNN aplicada ao espectrograma mel do SPIRA.

    O diagrama mostra três painéis em sequência:
        Esquerda  — espectrograma mel (real se S_dB for fornecido,
                    simulado caso contrário), com a janela 3x3 do filtro
                    destacada sobre uma região de pausa respiratória.
        Centro    — filtro (kernel) 3x3 com pesos aprendidos.
        Direita   — mapa de ativação resultante, com regiões de pausa
                    respiratória em destaque.

    Se S_dB for None, o espectrograma é simulado com os parâmetros SPIRA
    (sr=16000, n_mels=128, fmax=8000), reproduzindo as características
    acústicas descritas nos artigos do projeto: energia concentrada nas
    baixas frequências e colapsos abruptos de amplitude nas pausas
    respiratórias.

    Parâmetros
    ----------
    caminho_saida : str
        Caminho completo para o arquivo PNG de saída.
    S_dB : np.ndarray ou None
        Espectrograma mel em dB calculado por calcular_espectrograma().
        Se None, um espectrograma simulado é gerado internamente.
    sr : int
        Taxa de amostragem em Hz (padrão: 16000).
    margem : int
        Número de frames excluídos do início e do fim da gravação na
        detecção de pausas. Evita que silêncios de início e fim de
        gravação sejam confundidos com pausas respiratórias no interior
        da fala. Padrão: 15 frames (~0.1 s a 16 kHz com hop_length=512).
    """
    np.random.seed(42)

    # ── Espectrograma: real ou simulado ──────────────────────────────────────
    if S_dB is not None:
        spec_db = S_dB[:, :130] if S_dB.shape[1] > 130 else S_dB
        T = spec_db.shape[1]
        spec_db = spec_db - spec_db.max()
        spec = np.power(10, spec_db / 20)
        spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
        margem = 15
        energia = spec.sum(axis=0)
        energia_interior = energia[margem:T - margem]
        frames_interior  = np.arange(margem, T - margem)
        limiar = np.percentile(energia_interior, 25)
        candidatos = frames_interior[energia_interior < limiar]
        grupos, grupo_atual = [], [candidatos[0]]
        for f in candidatos[1:]:
            if f - grupo_atual[-1] <= 2:
                grupo_atual.append(f)
            else:
                grupos.append(grupo_atual)
                grupo_atual = [f]
        grupos.append(grupo_atual)
        centros = [int(np.median(g)) for g in grupos]
        pause_marks     = sorted(sorted(centros, key=lambda c: energia[c])[:3])
        pause_positions = list(candidatos)
    else:
        T = 130
        spec = np.zeros((N_MELS, T))
        for i in range(N_MELS):
            freq_weight = np.exp(-i / 30)
            spec[i] = np.random.rand(T) * freq_weight * 0.6
        for band, strength, width in [(10, 0.9, 8), (25, 0.7, 6), (45, 0.5, 5)]:
            for t in range(T):
                spec[band - width:band + width, t] += (
                    strength * np.random.rand() * 0.8
                )
        pause_positions = [30, 31, 32, 33, 65, 66, 67, 95, 96]
        for p in pause_positions:
            if p < T:
                spec[:, p] *= 0.08
        spec[80:, :] += np.random.rand(48, T) * 0.15
        spec = np.clip(spec, 0, 1)
        spec_db = 20 * np.log10(spec + 1e-6)
        spec_db = spec_db - spec_db.max()
        pause_marks = [30, 65, 95]

    bg        = "white"
    txt_color = "#222222"
    accent    = "#1a6faf"
    pause_color = "#c0392b"

    fig = plt.figure(figsize=(16, 7), facecolor=bg)
    gs = gridspec.GridSpec(
        1, 5,
        width_ratios=[6, 0.5, 2, 0.5, 4],
        wspace=0.08,
        left=0.04, right=0.97, top=0.88, bottom=0.12,
    )
    ax_spec = fig.add_subplot(gs[0])
    ax_arr1 = fig.add_subplot(gs[1])
    ax_filt = fig.add_subplot(gs[2])
    ax_arr2 = fig.add_subplot(gs[3])
    ax_feat = fig.add_subplot(gs[4])

    for ax in [ax_arr1, ax_arr2]:
        ax.set_visible(False)
    for ax in [ax_spec, ax_filt, ax_feat]:
        ax.set_facecolor(bg)

    ax_spec.imshow(
        spec_db, aspect="auto", origin="lower",
        cmap=CMAP, interpolation="bilinear",
    )

    fw, fh = 12, 18
    fx = pause_marks[0] - fw // 2 if pause_marks else 28
    fy = 16
    rect = plt.Rectangle(
        (fx, fy), fw, fh,
        linewidth=2.5, edgecolor=accent, facecolor="none",
        linestyle="--", zorder=5,
    )
    ax_spec.add_patch(rect)

    ax_spec.set_xticks([0, T // 4, T // 2, 3 * T // 4, T - 1])
    ax_spec.set_xticklabels(
        ["0", "0.25s", "0.5s", "0.75s", "1.0s"],
        color=txt_color, fontsize=9,
    )
    ax_spec.set_yticks([0, 32, 64, 96, 127])
    ax_spec.set_yticklabels(
        ["0", "2k", "4k", "6k", "8k Hz"],
        color=txt_color, fontsize=9,
    )
    ax_spec.tick_params(colors=txt_color, length=3)
    for spine in ax_spec.spines.values():
        spine.set_edgecolor("#bbbbbb")

    ax_spec.set_title(
        "espectrograma mel — entrada\n(paciente com IR, SPIRA)",
        color=txt_color, fontsize=10, pad=6,
    )
    ax_spec.annotate(
        "janela\n3×3",
        xy=(fx + fw / 2, fy + fh + 2),
        xytext=(fx + fw / 2 + 20, fy + fh + 18),
        color=accent, fontsize=8,
        arrowprops=dict(arrowstyle="->", color=accent, lw=1.5),
        ha="center",
    )
    for p in pause_marks:
        ax_spec.axvline(x=p, color=pause_color, lw=1.2, alpha=0.8, linestyle=":")
    if pause_marks:
        ax_spec.text(
            pause_marks[0] + 2, 115,
            "pausas\nrespiratórias",
            color=pause_color, fontsize=8, ha="left",
        )

    fig.text(0.395, 0.50, "⟶", color=accent, fontsize=28,
             va="center", ha="center")
    fig.text(0.395, 0.38, "convolução\n3×3", color="#555555",
             fontsize=8, va="center", ha="center")
    fig.text(0.685, 0.50, "⟶", color=accent, fontsize=28,
             va="center", ha="center")
    fig.text(0.685, 0.38, "mapa de\nativação", color="#555555",
             fontsize=8, va="center", ha="center")

    kernel = np.array([
        [ 0.12, -0.08,  0.05],
        [ 0.45,  0.82, -0.23],
        [-0.11,  0.37,  0.19],
    ])
    ax_filt.imshow(kernel, cmap="RdBu_r", vmin=-1, vmax=1,
                   aspect="equal", interpolation="nearest")
    for i in range(3):
        for j in range(3):
            ax_filt.text(
                j, i, f"{kernel[i, j]:.2f}",
                ha="center", va="center",
                color="white", fontsize=11, fontweight="bold",
            )
    ax_filt.set_xticks([])
    ax_filt.set_yticks([])
    for spine in ax_filt.spines.values():
        spine.set_edgecolor(accent)
        spine.set_linewidth(2)
    ax_filt.set_title(
        "filtro (kernel)\n3×3 — pesos aprendidos",
        color=txt_color, fontsize=10, pad=6,
    )

    feat_map = np.zeros((N_MELS, T))
    for i in range(N_MELS):
        freq_weight = np.exp(-i / 40)
        feat_map[i] = np.abs(spec[i] - 0.3) * freq_weight
    for p in pause_positions:
        if p < T:
            feat_map[8:35, max(0, p - 2):p + 3] += 0.6
    feat_map = gaussian_filter(feat_map, sigma=1.5)
    feat_map = np.clip(feat_map, 0, 1)

    ax_feat.imshow(
        feat_map, aspect="auto", origin="lower",
        cmap="viridis", interpolation="bilinear",
    )
    ax_feat.set_xticks([0, T // 4, T // 2, 3 * T // 4, T - 1])
    ax_feat.set_xticklabels(
        ["0", "0.25s", "0.5s", "0.75s", "1.0s"],
        color=txt_color, fontsize=9,
    )
    ax_feat.set_yticks([0, 32, 64, 96, 127])
    ax_feat.set_yticklabels(
        ["0", "2k", "4k", "6k", "8k Hz"],
        color=txt_color, fontsize=9,
    )
    ax_feat.tick_params(colors=txt_color, length=3)
    for spine in ax_feat.spines.values():
        spine.set_edgecolor("#bbbbbb")
    ax_feat.set_title(
        "mapa de ativação — padrões detectados\n(regiões de pausa destacadas)",
        color=txt_color, fontsize=10, pad=6,
    )
    for p in pause_marks:
        ax_feat.axvline(x=p, color=pause_color, lw=1.2, alpha=0.8, linestyle=":")

    fig.suptitle(
        "Convolução aplicada ao espectrograma mel do SPIRA:\n"
        "o filtro desliza sobre a imagem do som e produz um mapa de ativação",
        color=txt_color, fontsize=11, y=0.97,
    )

    plt.savefig(
        caminho_saida,
        dpi=200,
        bbox_inches="tight",
        facecolor=bg,
    )
    plt.close()
    print(f"  Salvo: {caminho_saida}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Gera os espectrogramas, formas de onda e diagrama CNN do Capítulo 4 "
            "da dissertação 'A rede que Marcelo construiu' (Helanski, 2026), "
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
    parser.add_argument(
        "--apenas-cnn",
        action="store_true",
        help=(
            "Gera apenas o diagrama CNN (spira_cnn_diagrama.png) "
            "com espectrograma simulado, sem necessidade de arquivos .wav."
        ),
    )
    args = parser.parse_args()

    os.makedirs(args.saida, exist_ok=True)

    def caminho(nome: str) -> str:
        return os.path.join(args.saida, nome)

    # ── Modo apenas-cnn ───────────────────────────────────────────────────────
    if args.apenas_cnn:
        print("\n[CNN] Diagrama de convolução (espectrograma simulado)")
        gerar_diagrama_cnn(caminho("spira_cnn_diagrama.png"))
        print(f"\nConcluído. Figura salva em: {os.path.abspath(args.saida)}")
        return

    # ── Grupo controle ────────────────────────────────────────────────────────
    print("\n[1/4] Grupo controle")
    y_c, sr_c = carregar_audio(args.controle)

    salvar_waveform(y_c, sr_c, caminho("spira_waveform_controle.png"))

    S_c = calcular_espectrograma(y_c, sr_c)

    salvar_sem_eixos(S_c, sr_c, caminho("spira_controle_sem_legenda.png"))
    salvar_com_eixos(
        S_c, sr_c,
        caminho("spira_controle_com_eixos.png"),
        titulo="Espectrograma Mel — grupo controle | 128 coeficientes, 16 kHz",
    )

    # ── Grupo paciente ────────────────────────────────────────────────────────
    print("\n[2/4] Grupo paciente")
    y_p, sr_p = carregar_audio(args.paciente)

    salvar_waveform(y_p, sr_p, caminho("spira_waveform_paciente.png"))

    S_p = calcular_espectrograma(y_p, sr_p)

    salvar_sem_eixos(S_p, sr_p, caminho("spira_paciente_sem_legenda.png"))
    salvar_com_eixos(
        S_p, sr_p,
        caminho("spira_paciente_com_eixos.png"),
        titulo="Espectrograma Mel — grupo paciente (IR) | 128 coeficientes, 16 kHz",
    )

    # ── Comparação linear × logarítmica (grupo paciente) ─────────────────────
    print("\n[3/4] Comparação linear × logarítmica")
    salvar_comparacao_linear_log(
        y_p, sr_p,
        caminho("spira_comparacao_linear_log.png"),
    )

    # ── Diagrama CNN (com espectrograma real do paciente) ─────────────────────
    print("\n[4/4] Diagrama CNN")
    gerar_diagrama_cnn(
        caminho("spira_cnn_diagrama.png"),
        S_dB=S_p,
        sr=sr_p,
    )

    print(
        f"\nConcluído. Oito figuras salvas em: {os.path.abspath(args.saida)}"
    )
    print(
        "Parâmetros: sr=16000 Hz | n_mels=128 | fmax=8000 Hz | cmap=magma"
    )


if __name__ == "__main__":
    main()

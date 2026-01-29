# app.py — Mini-Difusão (MNIST) — Demo Didática (PC Brasília)
# -----------------------------------------------------------
# Este app espera um checkpoint .pth gerado no Colab contendo:
#  - model_state (state_dict do modelo)
#  - opcionalmente: T, beta_start, beta_end, time_dim, base, image_size, channels
#
# Se você salvar um .pth com outra arquitetura (ex: SimpleUNet diferente),
# vai dar "Missing keys / Unexpected keys". Este app tenta inferir parâmetros
# e dá feedback claro quando não bater.

import io
import math
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- UI ----------------
st.set_page_config(page_title="Mini-Difusão (MNIST) — PC Brasília", layout="wide")
st.title("Mini-Difusão (MNIST) — Demo Didática")
st.caption("App educacional: visualiza a geração por difusão (ruído → imagem).")

# ---------------- Utils ----------------
def seed_everything(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def is_probably_state_dict(d: dict) -> bool:
    if not isinstance(d, dict) or len(d) == 0:
        return False
    # Heurística: state_dict costuma ter muitos tensores e chaves com ".weight"/".bias"
    tensor_count = sum([torch.is_tensor(v) for v in d.values()])
    key_hits = sum([(".weight" in k or ".bias" in k or "running_" in k) for k in d.keys()])
    return tensor_count >= max(5, len(d) // 4) and key_hits >= max(3, len(d) // 6)

def sinusoidal_time_embedding(t, dim=64):
    device = t.device
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=device).float() / (half - 1))
    args = t.float().view(-1, 1) * freqs.view(1, -1)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

# ---------------- Model (MiniUNet didático) ----------------
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.gn1 = nn.GroupNorm(8, out_ch)
        self.gn2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(x); h = self.gn1(h); h = self.act(h)
        t = self.time_mlp(t_emb).view(-1, h.size(1), 1, 1)
        h = h + t
        h = self.conv2(h); h = self.gn2(h); h = self.act(h)
        return h + self.res_conv(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.block = ResidualBlock(in_ch, out_ch, time_dim)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x, t_emb):
        h = self.block(x, t_emb)
        return self.pool(h), h

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.block = ResidualBlock(in_ch, out_ch, time_dim)

    def forward(self, x, skip, t_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x, t_emb)

class MiniUNet(nn.Module):
    def __init__(self, time_dim=64, base=32, in_channels=1, out_channels=1):
        super().__init__()
        self.time_dim = time_dim
        self.base = base

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        self.in_conv = nn.Conv2d(in_channels, base, 3, padding=1)
        self.down1 = Down(base, base * 2, time_dim)
        self.down2 = Down(base * 2, base * 4, time_dim)

        self.bot1 = ResidualBlock(base * 4, base * 4, time_dim)
        self.bot2 = ResidualBlock(base * 4, base * 4, time_dim)

        self.up2 = Up(base * 4 + base * 4, base * 2, time_dim)
        self.up1 = Up(base * 2 + base * 2, base, time_dim)

        self.out_conv = nn.Conv2d(base, out_channels, 3, padding=1)

    def forward(self, x, t):
        t_emb = sinusoidal_time_embedding(t, dim=self.time_dim)
        t_emb = self.time_mlp(t_emb)

        x = self.in_conv(x)
        x, s1 = self.down1(x, t_emb)
        x, s2 = self.down2(x, t_emb)
        x = self.bot1(x, t_emb)
        x = self.bot2(x, t_emb)
        x = self.up2(x, s2, t_emb)
        x = self.up1(x, s1, t_emb)
        return self.out_conv(x)

# ---------------- Diffusion helpers ----------------
@torch.no_grad()
def sample_ddpm(model, T, alpha, alpha_bar, n=16, img_size=28, channels=1, device="cpu"):
    model.eval()
    x = torch.randn(n, channels, img_size, img_size, device=device)
    frames = []

    for t_inv in range(T - 1, -1, -1):
        t = torch.full((n,), t_inv, device=device, dtype=torch.long)

        eps_pred = model(x, t)

        a_bar = alpha_bar[t].view(-1, 1, 1, 1)
        x0_hat = (x - torch.sqrt(1 - a_bar) * eps_pred) / torch.sqrt(a_bar)

        a = alpha[t].view(-1, 1, 1, 1)
        z = torch.randn_like(x) if t_inv > 0 else torch.zeros_like(x)

        x = torch.sqrt(a) * x0_hat + torch.sqrt(1 - a) * z

        # salva alguns “frames”
        if t_inv in [T-1, int(T*0.75), int(T*0.5), int(T*0.25), 0]:
            frames.append(x.detach().cpu().clone())

    return x.detach().cpu(), frames

def to_grid(x):
    # x: [N, C, H, W] em [-1,1]
    x = (x + 1) / 2
    x = x.clamp(0, 1)
    # concatena horizontalmente as primeiras N imagens
    grid = torch.cat([img for img in x], dim=2)
    if grid.size(0) == 1:
        grid = grid.squeeze(0)  # remove canal se C==1
    return grid.numpy()

# ---------------- Sidebar ----------------
st.sidebar.header("Configurações")
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.caption(f"Device: {device}")

seed = st.sidebar.number_input("Seed", min_value=0, max_value=10000, value=42, step=1)
n = st.sidebar.slider("N amostras", 4, 32, 16, step=4)

ckpt_file = st.sidebar.file_uploader("Envie o checkpoint (.pth) do notebook", type=["pth"])

seed_everything(int(seed))

if ckpt_file is None:
    st.info("Envie um checkpoint .pth para habilitar a geração.")
    st.stop()

# ---------------- Load checkpoint safely ----------------
try:
    buf = io.BytesIO(ckpt_file.read())
    ckpt = torch.load(buf, map_location=device)
except Exception as e:
    st.error(f"Falha ao ler o .pth. Erro: {e}")
    st.stop()

# Se o usuário subir um tensor .pth (não é modelo)
if torch.is_tensor(ckpt):
    st.error(
        "Esse .pth parece ser um TENSOR (imagem/array), não um checkpoint de MODELO.\n\n"
        "Você precisa subir o .pth gerado com `torch.save({'model_state': model.state_dict(), ...}, 'arquivo.pth')`."
    )
    st.stop()

# Determina state_dict e metadados
meta = {}
state_dict = None

if isinstance(ckpt, dict):
    meta = ckpt
    if "model_state" in ckpt and isinstance(ckpt["model_state"], dict):
        state_dict = ckpt["model_state"]
    elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        state_dict = ckpt["state_dict"]
    elif is_probably_state_dict(ckpt):
        state_dict = ckpt  # o próprio dict já é state_dict

if state_dict is None:
    st.error(
        "Não consegui encontrar o state_dict dentro do .pth.\n\n"
        "Formatos aceitos:\n"
        "- {'model_state': model.state_dict(), ...}\n"
        "- {'state_dict': model.state_dict(), ...}\n"
        "- ou o próprio state_dict puro (dict com pesos)\n"
    )
    st.stop()

# Inferência de hiperparâmetros (se existirem no checkpoint)
T_default = int(meta.get("T", 200))
beta_start_default = float(meta.get("beta_start", 1e-4))
beta_end_default = float(meta.get("beta_end", 0.02))
time_dim_default = int(meta.get("time_dim", 64))
base_default = int(meta.get("base", 32))
img_size = int(meta.get("image_size", 28))
channels = int(meta.get("channels", meta.get("channel", 1)))

# Sidebar permite override (caso queira)
st.sidebar.subheader("Parâmetros (do checkpoint / override)")
T = st.sidebar.slider("Passos T", 50, 600, int(T_default), step=50)
beta_start = st.sidebar.number_input("beta_start", value=float(beta_start_default), format="%.6f")
beta_end = st.sidebar.number_input("beta_end", value=float(beta_end_default), format="%.4f")

time_dim = st.sidebar.selectbox("time_dim (modelo)", options=[32, 64, 128], index=[32,64,128].index(time_dim_default) if time_dim_default in [32,64,128] else 1)
base = st.sidebar.selectbox("base (largura)", options=[16, 32, 48, 64], index=[16,32,48,64].index(base_default) if base_default in [16,32,48,64] else 1)

# Recria schedules
beta = torch.linspace(float(beta_start), float(beta_end), int(T)).to(device)
alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

# ---------------- Build model according to inferred params ----------------
model = MiniUNet(time_dim=int(time_dim), base=int(base), in_channels=channels, out_channels=channels).to(device)

# Tenta carregar pesos com diagnóstico
try:
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
except Exception as e:
    st.error(f"Falha ao carregar state_dict. Erro: {e}")
    st.stop()

# Se for mismatch grande, avisa com instrução prática
if len(missing) > 0 or len(unexpected) > 0:
    st.warning(
        "Checkpoint carregou com *diferenças* (isso pode ser OK se forem poucas).\n\n"
        f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}"
    )
    with st.expander("Ver detalhes (missing/unexpected)"):
        st.write("Missing keys (exemplo):", missing[:50])
        st.write("Unexpected keys (exemplo):", unexpected[:50])

    # Heurística: se mismatch for muito grande, provavelmente arquitetura diferente
    if len(missing) > 20 or len(unexpected) > 20:
        st.error(
            "⚠️ Parece que o checkpoint foi salvo com uma ARQUITETURA diferente da usada no Streamlit.\n\n"
            "✅ Solução: gere o .pth no Colab usando o MESMO MiniUNet (mesmos parâmetros base/time_dim) e salvando assim:\n"
            "```python\n"
            "checkpoint = {\n"
            "  'model_state': model.state_dict(),\n"
            "  'T': T,\n"
            "  'beta_start': beta_start,\n"
            "  'beta_end': beta_end,\n"
            "  'time_dim': 64,\n"
            "  'base': 32,\n"
            "  'image_size': 28,\n"
            "  'channels': 1,\n"
            "}\n"
            "torch.save(checkpoint, 'mini_difusao_mnist.pth')\n"
            "```\n\n"
            "Depois suba esse arquivo aqui."
        )
        st.stop()

st.success("✅ Checkpoint carregado e compatível!")

# ---------------- Run sampling ----------------
colA, colB = st.columns([1, 1])

with colA:
    st.subheader("Gerar")
    go = st.button("Gerar amostras")

with colB:
    st.subheader("Metadados do checkpoint")
    st.json(
        {
            "T": int(T),
            "beta_start": float(beta_start),
            "beta_end": float(beta_end),
            "time_dim": int(time_dim),
            "base": int(base),
            "image_size": int(img_size),
            "channels": int(channels),
        }
    )

if go:
    with st.spinner("Gerando amostras por difusão reversa..."):
        samples, frames = sample_ddpm(
            model=model,
            T=int(T),
            alpha=alpha,
            alpha_bar=alpha_bar,
            n=int(n),
            img_size=int(img_size),
            channels=int(channels),
            device=device,
        )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Resultado final")
        show_n = min(int(n), 16)
        st.image(to_grid(samples[:show_n]), clamp=True)
    with c2:
        st.subheader("Evolução (ruído → imagem)")
        show_n = min(int(n), 16)
        for i, f in enumerate(frames):
            st.caption(f"Frame {i+1}/{len(frames)}")
            st.image(to_grid(f[:show_n]), clamp=True)

st.divider()
st.caption(
    "Dica: se der mismatch, gere o checkpoint no Colab com o MESMO MiniUNet (base/time_dim) e salve com 'model_state'."
)

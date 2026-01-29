import io
import math
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

st.set_page_config(page_title="Mini-Difusão (MNIST) — PC Brasília", layout="wide")
st.title("Mini-Difusão (MNIST) — Demo Didática")
st.caption("App educacional: visualiza a geração por difusão (ruído → imagem).")

def seed_everything(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def sinusoidal_time_embedding(t, dim=64):
    device = t.device
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=device).float() / (half - 1))
    args = t.float().view(-1, 1) * freqs.view(1, -1)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0,1))
    return emb

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
    def __init__(self, time_dim=64, base=32):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.in_conv = nn.Conv2d(1, base, 3, padding=1)
        self.down1 = Down(base, base*2, time_dim)
        self.down2 = Down(base*2, base*4, time_dim)
        self.bot1 = ResidualBlock(base*4, base*4, time_dim)
        self.bot2 = ResidualBlock(base*4, base*4, time_dim)
        self.up2 = Up(base*4 + base*4, base*2, time_dim)
        self.up1 = Up(base*2 + base*2, base, time_dim)
        self.out_conv = nn.Conv2d(base, 1, 3, padding=1)
    def forward(self, x, t):
        t_emb = sinusoidal_time_embedding(t, dim=self.time_dim)
        t_emb = self.time_mlp(t_emb)
        x = self.in_conv(x)
        x, s1 = self.down1(x, t_emb)
        x, s2 = self.down2(x, t_emb)
        x = self.bot1(x, t_emb); x = self.bot2(x, t_emb)
        x = self.up2(x, s2, t_emb)
        x = self.up1(x, s1, t_emb)
        return self.out_conv(x)

st.sidebar.header("Configurações")
seed = st.sidebar.number_input("Seed", min_value=0, max_value=10000, value=42, step=1)
n = st.sidebar.slider("N amostras", 4, 32, 16, step=4)
T = st.sidebar.slider("Passos T", 50, 400, 200, step=50)
beta_start = st.sidebar.number_input("beta_start", value=1e-4, format="%.6f")
beta_end = st.sidebar.number_input("beta_end", value=0.02, format="%.4f")
ckpt_file = st.sidebar.file_uploader("Checkpoint (.pth) do notebook", type=["pth"])

device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.caption(f"Device: {device}")
seed_everything(int(seed))

beta = torch.linspace(float(beta_start), float(beta_end), T).to(device)
alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

@torch.no_grad()
def sample(model, n=16):
    model.eval()
    x = torch.randn(n, 1, 28, 28, device=device)
    frames = []
    for t_inv in range(T-1, -1, -1):
        t = torch.full((n,), t_inv, device=device, dtype=torch.long)
        eps_pred = model(x, t)
        a_bar = alpha_bar[t].view(-1, 1, 1, 1)
        x0_hat = (x - torch.sqrt(1 - a_bar) * eps_pred) / torch.sqrt(a_bar)
        a = alpha[t].view(-1, 1, 1, 1)
        z = torch.randn_like(x) if t_inv > 0 else torch.zeros_like(x)
        x = torch.sqrt(a) * x0_hat + torch.sqrt(1 - a) * z
        if t_inv in [T-1, int(T*0.75), int(T*0.5), int(T*0.25), 0]:
            frames.append(x.detach().cpu().clone())
    return x.detach().cpu(), frames

def to_grid(x):
    x = (x + 1) / 2
    x = x.clamp(0, 1)
    grid = torch.cat([img for img in x], dim=2).squeeze(0).numpy()
    return grid

model = MiniUNet(time_dim=64, base=32).to(device)

if ckpt_file is None:
    st.warning("Envie um checkpoint .pth (gerado no notebook) para habilitar a geração.")
    st.stop()

buf = io.BytesIO(ckpt_file.read())
ckpt = torch.load(buf, map_location=device)
model.load_state_dict(ckpt["model_state"])
st.success("Checkpoint carregado!")

if st.button("Gerar amostras"):
    samples, frames = sample(model, n=n)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Resultado final")
        st.image(to_grid(samples[:min(n,16)]), clamp=True)
    with c2:
        st.subheader("Evolução (ruído → imagem)")
        for i, f in enumerate(frames):
            st.caption(f"Frame {i+1}/{len(frames)}")
            st.image(to_grid(f[:min(n,16)]), clamp=True)


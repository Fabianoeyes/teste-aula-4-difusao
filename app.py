import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# MODELO (o MESMO do notebook)
# =====================================================

class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1)
        )

    def forward(self, x, t=None):
        return self.net(x)

# =====================================================
# UI
# =====================================================

st.set_page_config(layout="wide")
st.title("Mini-Difusão (MNIST) — Demo Didática")
st.caption("Upload de checkpoint treinado no notebook")

uploaded_file = st.file_uploader(
    "Envie o arquivo .pth do modelo (checkpoint)",
    type=["pth"]
)

# =====================================================
# FUNÇÃO DE AMOSTRAGEM SIMPLIFICADA
# =====================================================

def sample(model, n=16, image_size=28):
    model.eval()
    x = torch.randn(n, 1, image_size, image_size)
    with torch.no_grad():
        for _ in range(50):  # poucos passos para demo
            eps = model(x)
            x = x - 0.1 * eps
    return x

# =====================================================
# CARREGAMENTO DO CHECKPOINT
# =====================================================

if uploaded_file:
    try:
        ckpt = torch.load(uploaded_file, map_location="cpu")

        if not isinstance(ckpt, dict) or "model_state" not in ckpt:
            st.error("Arquivo inválido. Envie um checkpoint com 'model_state'.")
            st.stop()

        model = SimpleUNet()
        model.load_state_dict(ckpt["model_state"])

        T = ckpt.get("T", "N/A")
        beta_start = ckpt.get("beta_start", "N/A")
        beta_end = ckpt.get("beta_end", "N/A")

        st.success("Checkpoint carregado com sucesso!")

        col1, col2, col3 = st.columns(3)
        col1.metric("Passos T", T)
        col2.metric("Beta início", beta_start)
        col3.metric("Beta fim", beta_end)

        # =================================================
        # GERAR IMAGENS
        # =================================================
        st.subheader("Amostras geradas por difusão reversa")

        samples = sample(model)

        fig, axes = plt.subplots(4, 4, figsize=(6, 6))
        for ax, img in zip(axes.flatten(), samples):
            ax.imshow(img.squeeze(), cmap="gray")
            ax.axis("off")

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")

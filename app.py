# --- carregar checkpoint de forma robusta ---
buf = io.BytesIO(ckpt_file.read())
ckpt = torch.load(buf, map_location=device)

# checkpoint pode vir como dict com model_state ou como state_dict direto
if isinstance(ckpt, dict) and "model_state" in ckpt:
    state = ckpt["model_state"]
    ckpt_T = ckpt.get("T", None)
    ckpt_beta_start = ckpt.get("beta_start", None)
    ckpt_beta_end = ckpt.get("beta_end", None)
else:
    state = ckpt
    ckpt_T = ckpt_beta_start = ckpt_beta_end = None

# (opcional) se quiser travar a config do app igual à do notebook
# se existir no checkpoint, sobrescreve os sliders
if ckpt_T is not None:
    T = int(ckpt_T)
if ckpt_beta_start is not None:
    beta_start = float(ckpt_beta_start)
if ckpt_beta_end is not None:
    beta_end = float(ckpt_beta_end)

# recria schedules com os valores finais
beta = torch.linspace(float(beta_start), float(beta_end), T).to(device)
alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

# carrega pesos (aqui precisa bater 100% se a arquitetura for a mesma do notebook)
model.load_state_dict(state, strict=True)

st.success("Checkpoint carregado com sucesso!")

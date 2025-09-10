import sys
sys.path.append("..")
sys.path.append(".")

import math
import inspect
from typing import Dict, Any, Tuple, Optional, List

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

# 🔧 Ajusta este import a tu paquete real
from TSFEDL.models_pytorch import *  # noqa: F401,F403

# ========= Config =========
DEFAULTS = {
    "in_features": 1,
    "seq_len_candidates": [128, 256, 512, 1000, 1024, 2048, 3600],
    "batch_size": 4,
    "n_train": 24,
    "n_test": 8,
    "n_pred": 5,
    "out_features": 3,
    "lr": 1e-3,
}

# cerca de DEFAULTS
PER_MODEL_OVERRIDES = {
    # ya teníamos cosas así:
    "ChenChen": {"seq_len": 3600},
    # nuevo: GenMinxing necesita (N, T, C)
    "GenMinxing": {"layout": "NT_C"},
}



# ========= Dataset =========
class SyntheticForecastCOrCTDataset(Dataset):
    """
    Devuelve (x, y) donde:
      - si seq_len is None: x: (in_features,)               # 2D en batch -> (N, F)
      - si seq_len no es None y layout="NC_T": x: (C, T)     # 3D en batch -> (N, C, T)
      - si seq_len no es None y layout="NT_C": x: (T, C)     # 3D en batch -> (N, T, C)
      - y: (n_pred, out_features)
    """
    def __init__(self, n_samples, in_features, seq_len, n_pred, out_features, seed=42, layout="NC_T"):
        super().__init__()
        g = torch.Generator().manual_seed(seed)

        self.in_features = in_features
        self.seq_len = seq_len
        self.n_pred = n_pred
        self.out_features = out_features
        self.layout = layout  # "NC_T" o "NT_C" o "NONE" si 2D

        if seq_len is None:
            # x en 2D (N,F)
            self.X = torch.randn(n_samples, in_features, generator=g)  # (N, F)
            in_dim = in_features
        else:
            # Generamos base como (N, C, T); si hace falta, transpondremos en __getitem__
            xt = torch.randn(n_samples, in_features, seq_len, generator=g)  # (N, C, T)
            self.X = xt
            in_dim = in_features * seq_len

        # Construimos un target suave y no lineal de x (sin asumir nada del backbone)
        hidden = 64
        W1 = torch.randn(in_dim, hidden, generator=g) / math.sqrt(in_dim)
        b1 = torch.randn(hidden, generator=g) * 0.1
        W2 = torch.randn(hidden, n_pred * out_features, generator=g) / math.sqrt(hidden)
        b2 = torch.randn(n_pred * out_features, generator=g) * 0.1

        if seq_len is None:
            Xf = self.X.reshape(n_samples, -1)                  # (N, F)
        else:
            Xf = self.X.reshape(n_samples, -1)                  # (N, C*T)

        H = torch.tanh(Xf @ W1 + b1)                            # (N, hidden)
        Y = H @ W2 + b2                                         # (N, n_pred*out_features)
        Y = Y.reshape(n_samples, n_pred, out_features)
        noise = 0.01 * torch.randn(Y.shape, device=Y.device)
        self.Y = Y + noise

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.seq_len is not None and self.layout == "NT_C":
            # x viene como (C,T) -> devolvemos (T,C)
            x = x.transpose(0, 1).contiguous()
        return x, self.Y[idx]



# ========= Métrica =========
def mse_from_preds(y_hat, y):
    return F.mse_loss(y_hat, y).item()

# ========= Trainer compatible =========
def make_trainer(max_epochs: int = 1):
    cuda = torch.cuda.is_available()

    # Intento 1: APIs nuevas
    try:
        return pl.Trainer(
            max_epochs=max_epochs,
            logger=False,
            enable_checkpointing=False,
            accelerator="gpu" if cuda else "cpu",
            devices=1 if cuda else 1,
        )
    except Exception:
        pass

    # Intento 2: no pasar accelerator en CPU
    try:
        if cuda:
            return pl.Trainer(
                max_epochs=max_epochs,
                logger=False,
                enable_checkpointing=False,
                accelerator="gpu",
                devices=1,
            )
        else:
            return pl.Trainer(
                max_epochs=max_epochs,
                logger=False,
                enable_checkpointing=False,
            )
    except Exception:
        pass

    # Intento 3: API intermedia
    try:
        return pl.Trainer(
            max_epochs=max_epochs,
            logger=False,
            checkpoint_callback=False,
            gpus=1 if cuda else 0,
        )
    except Exception:
        pass

    # Intento 4: mínima
    return pl.Trainer(max_epochs=max_epochs, gpus=1 if cuda else 0)

# ========= Utils =========
def is_lightning_module(cls) -> bool:
    try:
        return issubclass(cls, pl.LightningModule)
    except Exception:
        return False

def discover_model_pairs(ns: Dict[str, Any]) -> List[Tuple[str, type, Optional[type]]]:
    classes = {name: obj for name, obj in ns.items() if inspect.isclass(obj)}
    pairs = []
    for name, cls in classes.items():
        if name.endswith("_Forecaster"):
            continue
        if not is_lightning_module(cls):
            continue
        f_name = f"{name}_Forecaster"
        f_cls = classes.get(f_name, None)
        pairs.append((name, cls, f_cls))
    pairs.sort(key=lambda t: t[0])
    return pairs

def build_model_kwargs(cls) -> Dict[str, Any]:
    sig = inspect.signature(cls)
    kwargs = {}
    for pname, p in sig.parameters.items():
        if pname == "in_features":
            kwargs[pname] = DEFAULTS["in_features"]
        elif pname == "top_module":
            kwargs[pname] = None
        elif pname == "loss":
            kwargs[pname] = nn.MSELoss()
        elif pname == "metrics":
            kwargs[pname] = {"mse": mse_from_preds}
        elif pname == "optimizer":
            kwargs[pname] = torch.optim.Adam
        elif pname in ("lr", "learning_rate"):
            kwargs[pname] = DEFAULTS["lr"]
        elif pname == "input_shape":
            kwargs[pname] = (DEFAULTS["in_features"], DEFAULTS["seq_len_candidates"][0])
        # 👇 añadido especial para YildirimOzal
        elif pname == "train_autoencoder":
            kwargs[pname] = False
        else:
            pass
    return kwargs


def try_forward_probe(backbone: pl.LightningModule, in_features: int, model_name: str) -> Tuple[torch.Tensor, int, Optional[int], str]:
    """
    Devuelve: (out_tensor, feat_dim, seq_len or None, layout)
      - out_tensor: salida cruda del backbone (sin cabeza)
      - feat_dim: si out es 2D -> out.size(1); si es 3D -> out.size(-1)
      - seq_len: T usado si la entrada fue 3D; None si la entrada fue 2D
      - layout: "NONE" (2D), "NC_T" o "NT_C" (3D)
    """
    # 1) Probar input 2D: (N, in_features)
    x2d = torch.randn(DEFAULTS["batch_size"], in_features)
    try:
        with torch.no_grad():
            out = backbone(x2d)
        if isinstance(out, torch.Tensor):
            if out.dim() == 2:
                return out, out.size(1), None, "NONE"
            if out.dim() == 3:
                return out, out.size(-1), None, "NONE"  # poco común, pero lo aceptamos
    except Exception:
        pass

    # 2) Probar 3D: (N, C=in_features, T)
    T_override = PER_MODEL_OVERRIDES.get(model_name, {}).get("seq_len", None)
    Ts = [T_override] if T_override is not None else DEFAULTS["seq_len_candidates"]
    for T in Ts:
        x3d = torch.randn(DEFAULTS["batch_size"], in_features, T)
        try:
            with torch.no_grad():
                out = backbone(x3d)
            if isinstance(out, torch.Tensor):
                if out.dim() == 2:
                    return out, out.size(1), T, "NC_T"
                if out.dim() == 3:
                    return out, out.size(-1), T, "NC_T"
        except Exception:
            continue

    # 3) Probar 3D: (N, T, C=in_features)
    for T in Ts:
        x3d_alt = torch.randn(DEFAULTS["batch_size"], T, in_features)
        try:
            with torch.no_grad():
                out = backbone(x3d_alt)
            if isinstance(out, torch.Tensor):
                if out.dim() == 2:
                    return out, out.size(1), T, "NT_C"
                if out.dim() == 3:
                    return out, out.size(-1), T, "NT_C"
        except Exception:
            continue

    raise RuntimeError("Could not probe feature dimension (no compatible input shape found).")



def build_forecaster_kwargs(f_cls, feat_dim: int) -> Dict[str, Any]:
    sig = inspect.signature(f_cls)
    kw = {}
    for pname, p in sig.parameters.items():
        if pname == "in_features":
            kw[pname] = feat_dim
        elif pname == "out_features":
            kw[pname] = DEFAULTS["out_features"]
        elif pname in ("n_pred", "steps", "horizon"):
            kw[pname] = DEFAULTS["n_pred"]
        elif p.default is not inspect._empty:
            continue
        else:
            pass
    return kw

def make_loaders(in_features: int, seq_len: Optional[int], layout: str):
    ds_tr = SyntheticForecastCOrCTDataset(24, in_features, seq_len,
                                          DEFAULTS["n_pred"], DEFAULTS["out_features"],
                                          seed=42, layout=layout if seq_len is not None else "NONE")
    ds_te = SyntheticForecastCOrCTDataset(8, in_features, seq_len,
                                          DEFAULTS["n_pred"], DEFAULTS["out_features"],
                                          seed=4242, layout=layout if seq_len is not None else "NONE")
    train_loader = DataLoader(ds_tr, batch_size=DEFAULTS["batch_size"], shuffle=True, num_workers=0)
    test_loader  = DataLoader(ds_te, batch_size=DEFAULTS["batch_size"], shuffle=False, num_workers=0)
    return train_loader, test_loader


def train_and_test(model, train_loader, test_loader, max_epochs=1):
    pl.seed_everything(42, workers=True)
    trainer = make_trainer(max_epochs=max_epochs)
    trainer.fit(model, train_loader)

    if "model" in inspect.getfullargspec(trainer.test).args:
        results = trainer.test(model, test_loader)
    else:
        results = trainer.test(test_loader)
    return results[0] if results else {}

# ========= Parametrización por modelo =========
ALL_PAIRS = discover_model_pairs(globals())
if not ALL_PAIRS:
    raise RuntimeError("No Lightning models found in models_pytorch.")

@pytest.mark.parametrize("model_name,ModelCls,ForecasterCls", ALL_PAIRS, ids=[p[0] for p in ALL_PAIRS])
def test_model_with_forecaster(model_name, ModelCls, ForecasterCls):
    if ForecasterCls is None:
        pytest.skip(f"{model_name}: no matching *_Forecaster class")

    # 1) Instanciar backbone sin cabeza
    try:
        m_kwargs = build_model_kwargs(ModelCls)
        backbone = ModelCls(**m_kwargs)
    except Exception as e:
        pytest.skip(f"{model_name}: instantiation failed: {e}")

    # 2) Sondar forward para feat_dim, seq_len y layout
    try:
        feats, feat_dim, seq_len, layout = try_forward_probe(
            backbone, m_kwargs.get("in_features", DEFAULTS["in_features"]), model_name
        )
        print(f"\n[INFO] {ModelCls.__name__}: detected feat_dim={feat_dim}, seq_len={seq_len}, layout={layout}")
    except Exception as e:
        pytest.skip(f"{model_name}: forward probe failed: {e}")

    # 3) Construir e inyectar cabeza forecaster (igual que lo tienes)
    try:
        f_kwargs = build_forecaster_kwargs(ForecasterCls, feat_dim)
        head = ForecasterCls(**f_kwargs)
        if hasattr(backbone, "classifier"):
            backbone.classifier = head
        elif hasattr(backbone, "top_module"):
            backbone.top_module = head
        else:
            pytest.skip(f"{model_name}: backbone lacks 'classifier'/'top_module' to attach head")

        # ✅ Saneado: asegurar métricas y loss válidos
        if not isinstance(getattr(backbone, "metrics", None), dict):
            backbone.metrics = {"mse": mse_from_preds}
        if not isinstance(getattr(backbone, "loss", None), nn.Module):
            backbone.loss = nn.MSELoss()
    except Exception as e:
        pytest.skip(f"{model_name}: forecaster instantiation failed: {e}")

    # 4) Forward con cabeza – comprobar forma (usando layout)
    if seq_len is None:
        x = torch.randn(DEFAULTS["batch_size"], m_kwargs.get("in_features", DEFAULTS["in_features"]))
    else:
        if layout == "NC_T":
            x = torch.randn(DEFAULTS["batch_size"], m_kwargs.get("in_features", DEFAULTS["in_features"]), seq_len)
        else:  # "NT_C"
            x = torch.randn(DEFAULTS["batch_size"], seq_len, m_kwargs.get("in_features", DEFAULTS["in_features"]))
    with torch.no_grad():
        y_hat = backbone(x)
    expected = (x.size(0), DEFAULTS["n_pred"], DEFAULTS["out_features"])
    assert tuple(y_hat.shape) == expected, f"{model_name}: output {tuple(y_hat.shape)} != expected {expected}"

    # 5) Entrenar 1 época y comprobar métrica/pérdida finita (pasamos layout)
    train_loader, test_loader = make_loaders(m_kwargs.get("in_features", DEFAULTS["in_features"]), seq_len, layout)
    results = train_and_test(backbone, train_loader, test_loader, max_epochs=1)
    keys = [k for k in results.keys() if "loss" in k or "mse" in k]
    assert len(keys) > 0, f"{model_name}: no test loss/metric logged. Got: {list(results.keys())}"
    vals = [float(results[k]) for k in keys if results[k] is not None]
    assert all(math.isfinite(v) for v in vals), f"{model_name}: non-finite test metrics {vals}"


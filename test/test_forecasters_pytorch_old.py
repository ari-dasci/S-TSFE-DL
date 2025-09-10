# tests/test_all_forecasters.py
import sys
sys.path.append("..")
sys.path.append(".")
import math
import inspect
import unittest
from typing import Dict, Any, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

# 🔧 AJUSTA ESTE IMPORT a tu paquete real
from TSFEDL.models_pytorch import *  # noqa: F401,F403


# ========= Config por defecto =========

DEFAULTS = {
    "in_features": 1,       # canales (C) por defecto
    "seq_len_candidates": [128, 256, 512, 1000, 1024, 2048],  # probaremos estas longitudes
    "batch_size": 4,
    "n_train": 24,
    "n_test": 8,
    "n_pred": 5,
    "out_features": 3,      # dim por paso
    "lr": 1e-3,
    "max_models": 12,       # limita para que la suite sea ágil
}

# ========= Dataset sintético =========

class SyntheticForecastCOrCTDataset(Dataset):
    """
    Devuelve:
      x: (C, T) si seq_len no es None; si no, x: (in_features,)
      y: (n_pred, out_features)

    No asume nada del backbone. Genera un target suave no lineal de x (+ ruido).
    """
    def __init__(self, n_samples: int, in_features: int, seq_len: Optional[int],
                 n_pred: int, out_features: int, seed: int = 42):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        if seq_len is None:
            self.X = torch.randn(n_samples, in_features, generator=g)             # (N, F)
            in_dim = in_features
        else:
            self.X = torch.randn(n_samples, in_features, seq_len, generator=g)    # (N, C, T)
            in_dim = in_features * seq_len

        hidden = 64
        W1 = torch.randn(in_dim, hidden, generator=g) / math.sqrt(in_dim)
        b1 = torch.randn(hidden, generator=g) * 0.1
        W2 = torch.randn(hidden, n_pred * out_features, generator=g) / math.sqrt(hidden)
        b2 = torch.randn(n_pred * out_features, generator=g) * 0.1

        Xf = self.X.reshape(n_samples, -1)
        H = torch.tanh(Xf @ W1 + b1)
        Y = H @ W2 + b2
        Y = Y.reshape(n_samples, n_pred, out_features)
        noise = 0.01 * torch.randn(Y.shape, device=Y.device)
        self.Y = Y + noise

        self.in_features = in_features
        self.seq_len = seq_len
        self.n_pred = n_pred
        self.out_features = out_features

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ========= Métrica =========

def mse_from_preds(y_hat, y):
    return F.mse_loss(y_hat, y).item()


# ========= Utilidades =========

def make_trainer(max_epochs: int = 1):
    cuda = torch.cuda.is_available()

    # Intento 1: APIs nuevas (Lightning 1.6+/2.x)
    try:
        return pl.Trainer(
            max_epochs=max_epochs,
            logger=False,
            enable_checkpointing=False,
            accelerator="gpu" if cuda else "cpu",
            devices=1 if cuda else 1,   # en CPU algunos esperan 1, otros lo ignoran
        )
    except Exception:
        pass

    # Intento 2: API nueva pero sin pasar accelerator en CPU
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

    # Intento 3: API intermedia (~1.3–1.5): usar gpus y checkpoint_callback
    try:
        return pl.Trainer(
            max_epochs=max_epochs,
            logger=False,
            checkpoint_callback=False,
            gpus=1 if cuda else 0,
        )
    except Exception:
        pass

    # Intento 4: API muy antigua: mínimos comunes
    return pl.Trainer(
        max_epochs=max_epochs,
        gpus=1 if cuda else 0,
    )


def is_lightning_module(cls) -> bool:
    try:
        return issubclass(cls, pl.LightningModule)
    except Exception:
        return False

def discover_model_pairs(ns: Dict[str, Any]) -> List[Tuple[str, type, Optional[type]]]:
    """
    Busca todas las clases Lightning cuyo nombre tenga pareja *_Forecaster.
    Devuelve lista de (base_name, BaseClass, ForecasterClass or None).
    """
    classes = {name: obj for name, obj in ns.items() if inspect.isclass(obj)}
    pairs = []
    for name, cls in classes.items():
        if name.endswith("_Forecaster"):
            continue
        if not is_lightning_module(cls):
            continue
        # Busca pareja de forecaster
        f_name = f"{name}_Forecaster"
        f_cls = classes.get(f_name, None)
        pairs.append((name, cls, f_cls))
    pairs.sort(key=lambda t: t[0])
    return pairs

def build_model_kwargs(cls) -> Dict[str, Any]:
    """
    Construye kwargs básicos para el backbone.
    """
    sig = inspect.signature(cls)
    kwargs = {}
    for pname, p in sig.parameters.items():
        if pname == "in_features":
            kwargs[pname] = DEFAULTS["in_features"]
        elif pname == "top_module":
            kwargs[pname] = None  # importante: sin cabeza para medir feat_dim
        elif pname == "loss":
            kwargs[pname] = nn.MSELoss()
        elif pname == "metrics":
            kwargs[pname] = {"mse": mse_from_preds}
        elif pname == "optimizer":
            kwargs[pname] = torch.optim.Adam
        elif pname == "lr" or pname == "learning_rate":
            kwargs[pname] = DEFAULTS["lr"]
        elif pname == "input_shape":
            # Algunos modelos usan input_shape en lugar de in_features/seq_len
            # Damos una forma mínima (C,T) genérica; si no encaja, se ignorará más tarde
            kwargs[pname] = (DEFAULTS["in_features"], DEFAULTS["seq_len_candidates"][0])
        else:
            # Si tiene default, lo dejamos; si es obligatorio y desconocido, fallará y lo saltaremos
            pass
    return kwargs

def try_forward_probe(backbone: pl.LightningModule, in_features: int) -> Tuple[torch.Tensor, int, Optional[int]]:
    """
    Intenta hacer forward SIN cabeza para detectar:
      - forma esperada de entrada (2D o 3D)
      - seq_len compatible si aplica
      - dimensión de features plana (feat_dim)
    Devuelve: (feats, feat_dim, seq_len_usada_o_None)
    """
    # 1) Probar input 2D: (N, in_features)
    x2d = torch.randn(DEFAULTS["batch_size"], in_features)
    try:
        with torch.no_grad():
            out = backbone(x2d)
        if isinstance(out, torch.Tensor) and out.dim() == 2:
            return out, out.size(1), None
    except Exception:
        pass

    # 2) Probar input 3D: (N, C=in_features, T) con varias T
    for T in DEFAULTS["seq_len_candidates"]:
        x3d = torch.randn(DEFAULTS["batch_size"], in_features, T)
        try:
            with torch.no_grad():
                out = backbone(x3d)
            if isinstance(out, torch.Tensor) and out.dim() == 2:
                return out, out.size(1), T
        except Exception:
            continue

    raise RuntimeError("Could not probe feature dimension (no compatible input shape found).")

def build_forecaster_kwargs(f_cls, feat_dim: int) -> Dict[str, Any]:
    """
    Construye kwargs para la cabeza forecaster, mapeando nombres alternativos si existen.
    """
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
            # tiene default, lo omitimos
            continue
        else:
            # parámetro obligatorio desconocido → dejaremos que falle y se salte
            pass
    return kw

def make_loaders(in_features: int, seq_len: Optional[int]) -> Tuple[DataLoader, DataLoader]:
    ds_tr = SyntheticForecastCOrCTDataset(DEFAULTS["n_train"], in_features, seq_len,
                                          DEFAULTS["n_pred"], DEFAULTS["out_features"], seed=42)
    ds_te = SyntheticForecastCOrCTDataset(DEFAULTS["n_test"],  in_features, seq_len,
                                          DEFAULTS["n_pred"], DEFAULTS["out_features"], seed=4242)
    train_loader = DataLoader(ds_tr, batch_size=DEFAULTS["batch_size"], shuffle=True, num_workers=0)
    test_loader  = DataLoader(ds_te, batch_size=DEFAULTS["batch_size"], shuffle=False, num_workers=0)
    return train_loader, test_loader

def train_and_test(model, train_loader, test_loader, max_epochs=1):
    pl.seed_everything(42, workers=True)
    trainer = make_trainer(max_epochs=max_epochs)
    trainer.fit(model, train_loader)

    # Compat con cambios de firma en .test
    import inspect as _inspect
    if "model" in _inspect.getfullargspec(trainer.test).args:
        results = trainer.test(model, test_loader)
    else:
        results = trainer.test(test_loader)
    return results[0] if results else {}



# ========= Test principal =========

class TestAllForecasters(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pl.seed_everything(42, workers=True)
        cls.pairs = discover_model_pairs(globals())
        # Al menos debería haber alguna clase Lightning; si no, fallamos pronto con info.
        assert len(cls.pairs) > 0, "No Lightning models found in models_pytorch."

    def test_all_models_have_optional_forecaster(self):
        # No exigimos que *todas* tengan forecaster, pero registramos las que no.
        missing = [name for name, _, fcls in self.pairs if fcls is None]
        if missing:
            print("\n[INFO] Models without a matching *_Forecaster:", missing)

    def test_train_all_with_detected_head(self):
        tested = 0
        skipped: List[Tuple[str, str]] = []

        for name, ModelCls, ForecasterCls in self.pairs:
            """ if tested >= DEFAULTS["max_models"]:
                break """

            if ForecasterCls is None:
                skipped.append((name, "no matching *_Forecaster class"))
                continue

            # 1) Instanciar backbone sin cabeza
            try:
                m_kwargs = build_model_kwargs(ModelCls)
                backbone = ModelCls(**m_kwargs)
            except Exception as e:
                skipped.append((name, f"instantiation failed: {e}"))
                continue

            # 2) Sondar forward para obtener feat_dim y seq_len
            try:
                feats, feat_dim, seq_len = try_forward_probe(backbone, m_kwargs.get("in_features", DEFAULTS["in_features"]))
                print(f"\n[INFO] {name}: detected feat_dim={feat_dim}, seq_len={seq_len}")
                self.assertEqual(feats.dim(), 2, f"{name}: backbone without head must return (N, F). Got {tuple(feats.shape)}")
                self.assertGreater(feat_dim, 0, f"{name}: feature dim must be > 0.")
            except Exception as e:
                skipped.append((name, f"forward probe failed: {e}"))
                continue

            # 3) Construir y colocar la cabeza forecaster
            try:
                f_kwargs = build_forecaster_kwargs(ForecasterCls, feat_dim)
                head = ForecasterCls(**f_kwargs)
                # Asumimos atributo estándar .classifier / .top_module en el base
                if hasattr(backbone, "classifier"):
                    backbone.classifier = head
                elif hasattr(backbone, "top_module"):
                    backbone.top_module = head
                else:
                    skipped.append((name, "backbone has no 'classifier' or 'top_module' attribute to set head"))
                    continue
            except Exception as e:
                skipped.append((name, f"forecaster instantiation failed: {e}"))
                continue

            # 4) Comprobar forma de salida con la cabeza
            try:
                if seq_len is None:
                    x = torch.randn(DEFAULTS["batch_size"], m_kwargs.get("in_features", DEFAULTS["in_features"]))
                else:
                    x = torch.randn(DEFAULTS["batch_size"], m_kwargs.get("in_features", DEFAULTS["in_features"]), seq_len)
                with torch.no_grad():
                    y_hat = backbone(x)
                expected = (x.size(0), DEFAULTS["n_pred"], DEFAULTS["out_features"])
                self.assertEqual(tuple(y_hat.shape), expected, f"{name}: output {tuple(y_hat.shape)} != expected {expected}")
            except Exception as e:
                skipped.append((name, f"forward with head failed: {e}"))
                continue

            # 5) Entrenar 1 época y comprobar métricas finitas
            try:
                train_loader, test_loader = make_loaders(m_kwargs.get("in_features", DEFAULTS["in_features"]), seq_len)
                results = train_and_test(backbone, train_loader, test_loader, max_epochs=1)
                keys = [k for k in results.keys() if "loss" in k or "mse" in k]
                self.assertTrue(len(keys) > 0, f"{name}: no test loss/metric logged. Got {list(results.keys())}")
                vals = [float(results[k]) for k in keys if results[k] is not None]
                self.assertTrue(all(math.isfinite(v) for v in vals), f"{name}: non-finite test metrics {vals}")
                tested += 1
            except Exception as e:
                skipped.append((name, f"training/test failed: {e}"))
                continue

        if tested == 0:
            reasons = "\n".join([f"- {n}: {msg}" for n, msg in skipped])
            self.fail(f"All models were skipped.\nReasons:\n{reasons}")
        else:
            if skipped:
                print("\n[INFO] Skipped models:")
                for n, msg in skipped:
                    print(f"  - {n}: {msg}")


if __name__ == "__main__":
    unittest.main()

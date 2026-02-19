from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

try:
    from scipy.optimize import curve_fit
except Exception:  # scipy optional
    curve_fit = None


@dataclass
class FitConfig:
    has_qp: bool = False
    has_tls1: bool = False
    has_tls2: bool = False
    adam_steps: int = 200
    adam_lr: float = 0.05


def _inverse_q_model(power_dbm: np.ndarray, temp_mk: np.ndarray, params: Dict[str, float], cfg: FitConfig) -> np.ndarray:
    qother = max(params["Qother"], 1.0)
    inv_q = np.full_like(power_dbm, 1.0 / qother, dtype=float)

    if cfg.has_qp:
        qqp = max(params.get("Qqp", 1.0), 1.0)
        inv_q += np.exp(-0.04 * power_dbm) / qqp

    if cfg.has_tls1:
        qtls1 = max(params.get("Qtls1", 1.0), 1.0)
        inv_q += (temp_mk / np.maximum(np.max(temp_mk), 1.0)) / qtls1

    if cfg.has_tls2:
        qtls2 = max(params.get("Qtls2", 1.0), 1.0)
        inv_q += np.exp(-0.02 * (power_dbm - np.min(power_dbm))) / qtls2

    return np.maximum(inv_q, 1e-16)


def qi_model(power_dbm: np.ndarray, temp_mk: np.ndarray, params: Dict[str, float], cfg: FitConfig) -> np.ndarray:
    return 1.0 / _inverse_q_model(power_dbm, temp_mk, params, cfg)


def _single_param_fit(x: np.ndarray, y: np.ndarray, model_func: Callable[[np.ndarray, float], np.ndarray], p0: float) -> float:
    if len(x) < 2:
        return float(p0)
    if curve_fit is not None:
        try:
            popt, _ = curve_fit(model_func, x, y, p0=[p0], maxfev=2000)
            return float(popt[0])
        except Exception:
            return float(p0)

    grid = np.geomspace(max(1.0, p0 / 20), p0 * 20, 128)
    losses = [np.mean((model_func(x, g) - y) ** 2) for g in grid]
    return float(grid[int(np.argmin(losses))])


def estimate_start_values(power_dbm: np.ndarray, temp_mk: np.ndarray, qi: np.ndarray, cfg: FitConfig) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]]]:
    qimax = float(np.max(qi))
    starts: Dict[str, float] = {"Qother": qimax}
    bounds: Dict[str, Tuple[float, float]] = {"Qother": (qimax, 10.0 * qimax)}

    if cfg.has_qp:
        max_power = np.max(power_dbm)
        mask = power_dbm == max_power
        qqp_start = _single_param_fit(temp_mk[mask], qi[mask], lambda t, qqp: qi_model(np.full_like(t, max_power), t, {"Qother": qimax, "Qqp": qqp}, FitConfig(has_qp=True)), qimax)
        starts["Qqp"] = qqp_start
        bounds["Qqp"] = (max(1.0, qqp_start / 20), qqp_start * 20)

    if cfg.has_tls1:
        min_temp = np.min(temp_mk)
        mask = temp_mk == min_temp
        qtls1_start = _single_param_fit(power_dbm[mask], qi[mask], lambda p, qtls1: qi_model(p, np.full_like(p, min_temp), {"Qother": qimax, "Qtls1": qtls1}, FitConfig(has_tls1=True)), qimax)
        starts["Qtls1"] = qtls1_start
        bounds["Qtls1"] = (max(1.0, qtls1_start / 20), qtls1_start * 20)

    if cfg.has_tls2:
        min_power = np.min(power_dbm)
        mask = power_dbm == min_power
        qtls2_start = _single_param_fit(temp_mk[mask], qi[mask], lambda t, qtls2: qi_model(np.full_like(t, min_power), t, {"Qother": qimax, "Qtls2": qtls2}, FitConfig(has_tls2=True)), qimax)
        starts["Qtls2"] = qtls2_start
        bounds["Qtls2"] = (max(1.0, qtls2_start / 20), qtls2_start * 20)

    return starts, bounds


def _mse(power_dbm: np.ndarray, temp_mk: np.ndarray, qi: np.ndarray, params: Dict[str, float], cfg: FitConfig) -> float:
    pred = qi_model(power_dbm, temp_mk, params, cfg)
    return float(np.mean((pred - qi) ** 2))


def adam_refine(power_dbm: np.ndarray, temp_mk: np.ndarray, qi: np.ndarray, starts: Dict[str, float], bounds: Dict[str, Tuple[float, float]], cfg: FitConfig) -> Dict[str, float]:
    keys = list(starts.keys())
    x = np.array([starts[k] for k in keys], dtype=float)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    def to_params(vec: np.ndarray) -> Dict[str, float]:
        out = {}
        for i, k in enumerate(keys):
            lo, hi = bounds[k]
            out[k] = float(np.clip(vec[i], lo, hi))
        return out

    best = to_params(x)
    best_loss = _mse(power_dbm, temp_mk, qi, best, cfg)

    for t in range(1, cfg.adam_steps + 1):
        params = to_params(x)
        grads = np.zeros_like(x)
        for i, k in enumerate(keys):
            step = max(1e-6, abs(x[i]) * 1e-4)
            plus = x.copy(); plus[i] += step
            minus = x.copy(); minus[i] -= step
            g = (_mse(power_dbm, temp_mk, qi, to_params(plus), cfg) - _mse(power_dbm, temp_mk, qi, to_params(minus), cfg)) / (2 * step)
            grads[i] = g

        m = beta1 * m + (1 - beta1) * grads
        v = beta2 * v + (1 - beta2) * (grads ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        x = x - cfg.adam_lr * m_hat / (np.sqrt(v_hat) + eps)

        current = to_params(x)
        loss = _mse(power_dbm, temp_mk, qi, current, cfg)
        if loss < best_loss:
            best_loss = loss
            best = current

    return best


def smooth_curve(power_dbm: np.ndarray, temp_mk: np.ndarray, params: Dict[str, float], cfg: FitConfig, points: int = 600):
    p = np.linspace(np.min(power_dbm), np.max(power_dbm), points)
    t = np.full_like(p, np.min(temp_mk))
    return p, qi_model(p, t, params, cfg)


def style_paper_axes(ax, title: str = "") -> None:
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, axis="both", alpha=0.25, linewidth=0.8)
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def export_plot_data(
    path: str | Path,
    series: Iterable[Tuple[str, np.ndarray, np.ndarray]],
    delimiter: str = ";",
    excel_friendly: bool = True,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[str] = [f"series{delimiter}x{delimiter}y"]
    for label, x, y in series:
        for xi, yi in zip(np.asarray(x).ravel(), np.asarray(y).ravel()):
            rows.append(f"{label}{delimiter}{xi}{delimiter}{yi}")

    encoding = "utf-8-sig" if excel_friendly else "utf-8"
    path.write_text("\n".join(rows), encoding=encoding)
    return path


def enable_editable_legend(fig, ax, label_store: Dict[str, str] | None = None) -> Dict[str, str]:
    """Double-click or right-click on a legend label to rename it (kept in memory for runtime)."""
    if label_store is None:
        label_store = {}

    legend = ax.get_legend()
    if legend is None:
        legend = ax.legend(loc="best", frameon=False)

    lines = ax.get_lines()
    for line in lines:
        old = line.get_label()
        if old in label_store:
            line.set_label(label_store[old])
    legend = ax.legend(loc="best", frameon=False)

    def _on_click(event):
        if event.inaxes != ax:
            return
        if not (event.dblclick or event.button == 3):
            return
        leg = ax.get_legend()
        if leg is None:
            return
        for text in leg.get_texts():
            bbox = text.get_window_extent(renderer=fig.canvas.get_renderer())
            if bbox.contains(event.x, event.y):
                current = text.get_text()
                try:
                    new_label = input(f"Neuer Name für '{current}': ").strip()
                except Exception:
                    return
                if not new_label:
                    return
                label_store[current] = new_label
                for line in lines:
                    if line.get_label() == current:
                        line.set_label(new_label)
                ax.legend(loc="best", frameon=False)
                fig.canvas.draw_idle()
                return

    fig.canvas.mpl_connect("button_press_event", _on_click)
    return label_store

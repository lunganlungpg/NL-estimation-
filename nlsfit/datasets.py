"""Bundled datasets: NIST certified benchmarks, a classic real dataset, and
clearly labelled simulated datasets with known true parameters.

NIST StRD data and certified values are works of the US Government and are in
the public domain. Source: https://www.itl.nist.gov/div898/strd/nls/nls_main.shtml
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import models as M


@dataclass(frozen=True)
class Dataset:
    name: str
    x: np.ndarray
    y: np.ndarray
    model: M.Model
    description: str
    source: str
    x_label: str = "x"
    y_label: str = "y"
    kind: str = "real"  # "nist", "real" or "simulated"
    difficulty: str = ""
    starts: tuple[tuple[float, ...], ...] = ()
    certified: tuple[float, ...] | None = None
    certified_sse: float | None = None
    true_params: tuple[float, ...] | None = None
    notes: dict = field(default_factory=dict)

    def as_frame(self):
        import pandas as pd

        return pd.DataFrame({self.x_label: self.x, self.y_label: self.y})


def _a(*v):
    return np.array(v, dtype=float)


_NIST = "NIST Statistical Reference Datasets (public domain)"

_REGISTRY: dict[str, Dataset] = {}


def _register(ds: Dataset) -> Dataset:
    _REGISTRY[ds.name] = ds
    return ds


_register(
    Dataset(
        "misra1a",
        _a(77.6, 114.9, 141.1, 190.8, 239.9, 289.0, 332.8, 378.4, 434.8, 477.3, 536.8, 593.1, 689.1, 760.0),
        _a(10.07, 14.73, 17.94, 23.93, 29.61, 35.18, 40.02, 44.82, 50.76, 55.05, 61.01, 66.40, 75.47, 81.78),
        M.exponential_rise,
        "Dental research: monomolecular adsorption (Misra, 1978).",
        _NIST,
        "pressure",
        "volume",
        "nist",
        "lower",
        starts=((500, 1e-4), (250, 5e-4)),
        certified=(2.3894212918e02, 5.5015643181e-04),
        certified_sse=1.2455138894e-01,
    )
)

_register(
    Dataset(
        "boxbod",
        _a(1, 2, 3, 5, 7, 10),
        _a(109, 149, 149, 191, 213, 224),
        M.exponential_rise,
        "Biochemical oxygen demand (Box et al., 1978). Only 6 points; a classic trap for Gauss–Newton.",
        _NIST,
        "incubation time (days)",
        "BOD (mg/L)",
        "nist",
        "higher",
        starts=((1, 1), (100, 0.75)),
        certified=(2.1380940889e02, 5.4723748542e-01),
        certified_sse=1.1680088766e03,
    )
)

_register(
    Dataset(
        "danwood",
        _a(1.309, 1.471, 1.490, 1.565, 1.611, 1.680),
        _a(2.138, 3.421, 3.597, 4.340, 4.882, 5.660),
        M.power_law,
        "Radiated energy vs temperature (Daniel & Wood, 1980).",
        _NIST,
        "temperature (1000 K)",
        "energy",
        "nist",
        "lower",
        starts=((1, 5), (0.7, 4)),
        certified=(7.6886226176e-01, 3.8604055871e00),
        certified_sse=4.3173084083e-03,
    )
)

_register(
    Dataset(
        "rat42",
        _a(9, 14, 21, 28, 42, 57, 63, 70, 79),
        _a(8.93, 10.8, 18.59, 22.33, 39.35, 56.11, 61.73, 64.62, 67.08),
        M.logistic3,
        "Pasture yield vs growing time, logistic growth (Ratkowsky, 1983).",
        _NIST,
        "growing time (days)",
        "pasture yield",
        "nist",
        "higher",
        starts=((100, 1, 0.1), (75, 2.5, 0.07)),
        certified=(7.2462237576e01, 2.6180768402e00, 6.7359200066e-02),
        certified_sse=8.0565229338e00,
    )
)

_register(
    Dataset(
        "rat43",
        np.arange(1.0, 16.0),
        _a(
            16.08,
            33.83,
            65.80,
            97.20,
            191.55,
            326.20,
            386.87,
            520.53,
            590.03,
            651.92,
            724.93,
            699.56,
            689.96,
            637.56,
            717.41,
        ),
        M.richards,
        "Dry weight of onion bulbs vs growing time, Richards curve (Ratkowsky, 1983).",
        _NIST,
        "growing time",
        "dry weight",
        "nist",
        "higher",
        starts=((100, 10, 1, 1), (700, 5, 0.75, 1.3)),
        certified=(6.9964151270e02, 5.2771253025e00, 7.5962938329e-01, 1.2792483859e00),
        certified_sse=8.7864049080e03,
    )
)

_register(
    Dataset(
        "mgh09",
        _a(4, 2, 1, 0.5, 0.25, 0.167, 0.125, 0.1, 0.0833, 0.0714, 0.0625),
        _a(
            1.957e-01,
            1.947e-01,
            1.735e-01,
            1.600e-01,
            8.440e-02,
            6.270e-02,
            4.560e-02,
            3.420e-02,
            3.230e-02,
            2.350e-02,
            2.460e-02,
        ),
        M.kowalik_osborne,
        "Kowalik & Osborne enzyme problem (Moré, Garbow & Hillstrom, 1981). Very hard from start 1.",
        _NIST,
        "x",
        "y",
        "nist",
        "higher",
        starts=((25, 39, 41.5, 39), (0.25, 0.39, 0.415, 0.39)),
        certified=(1.9280693458e-01, 1.9128232873e-01, 1.2305650693e-01, 1.3606233068e-01),
        certified_sse=3.0750560385e-04,
    )
)

_register(
    Dataset(
        "puromycin",
        _a(0.02, 0.02, 0.06, 0.06, 0.11, 0.11, 0.22, 0.22, 0.56, 0.56, 1.10, 1.10),
        _a(76, 47, 97, 107, 123, 139, 159, 152, 191, 201, 207, 200),
        M.michaelis_menten,
        "Reaction velocity of an enzyme treated with puromycin (Treloar, 1974; Bates & Watts, 1988).",
        "Treloar (1974), as published in Bates & Watts (1988) and R's datasets::Puromycin",
        "substrate concentration (ppm)",
        "velocity (counts/min²)",
        "real",
    )
)


# ---------------------------------------------------------------- simulated


def _simulate(model, x, theta, noise, seed, multiplicative=False):
    rng = np.random.default_rng(seed)
    mu = model(x, *theta)
    e = rng.normal(0, noise, size=x.size)
    y = mu * np.exp(e) if multiplicative else mu + e
    return np.round(y, 3)


_x_n = np.repeat(np.array([0, 40, 80, 120, 160, 200, 240.0]), 3)
_register(
    Dataset(
        "crop_yield_nitrogen",
        _x_n,
        _simulate(M.asymptotic_regression, _x_n, (9.5, 6.0, 0.012), 0.35, 11),
        M.asymptotic_regression,
        "SIMULATED wheat grain yield response to nitrogen fertiliser, 7 rates × 3 replicates. "
        "Mitscherlich diminishing-returns curve.",
        "Simulated by nlsfit (seed 11)",
        "nitrogen rate (kg N/ha)",
        "grain yield (t/ha)",
        "simulated",
        true_params=(9.5, 6.0, 0.012),
    )
)

_x_g = np.arange(0, 71, 5.0)
_register(
    Dataset(
        "plant_growth",
        _x_g,
        _simulate(M.gompertz, _x_g, (180.0, 6.0, 0.92), 4.0, 23),
        M.gompertz,
        "SIMULATED maize biomass over the season. Generated from a Gompertz curve, "
        "so a logistic fit should lose on AIC.",
        "Simulated by nlsfit (seed 23)",
        "days after sowing",
        "biomass (g/plant)",
        "simulated",
        true_params=(180.0, 6.0, 0.92),
    )
)

_x_d = np.repeat(np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100.0]), 2)
_register(
    Dataset(
        "dose_response",
        _x_d,
        _simulate(M.four_pl, _x_d, (5.0, 98.0, 1.5, 1.2), 3.0, 5),
        M.four_pl,
        "SIMULATED % cell viability vs drug concentration, duplicate wells. Four-parameter logistic.",
        "Simulated by nlsfit (seed 5)",
        "concentration (µM)",
        "viability (%)",
        "simulated",
        true_params=(5.0, 98.0, 1.5, 1.2),
    )
)

_x_t = np.array([0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12])
_register(
    Dataset(
        "drug_elimination",
        _x_t,
        _simulate(M.exponential_decay, _x_t, (12.0, 0.35), 0.06, 7, multiplicative=True),
        M.exponential_decay,
        "SIMULATED plasma drug concentration after an IV bolus (first-order elimination). "
        "Noise is multiplicative (≈6% CV), a good case for weighted fitting.",
        "Simulated by nlsfit (seed 7)",
        "time (h)",
        "concentration (mg/L)",
        "simulated",
        true_params=(12.0, 0.35),
    )
)


def load(name: str) -> Dataset:
    """Load a bundled dataset by name. See :func:`list_datasets`."""
    try:
        return _REGISTRY[name.lower()]
    except KeyError:
        raise KeyError(f"Unknown dataset {name!r}. Available: {', '.join(_REGISTRY)}") from None


def list_datasets() -> list[str]:
    return list(_REGISTRY)


def log_relative_error(estimate, certified) -> np.ndarray:
    """NIST's accuracy measure: number of correct significant digits (LRE).

    ``LRE = −log10(|estimate − certified| / |certified|)``, capped at 11.
    An LRE of 6 or more is generally considered a successful solve.
    """
    est, cert = np.asarray(estimate, float), np.asarray(certified, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lre = -np.log10(np.abs(est - cert) / np.abs(cert))
    return np.clip(np.nan_to_num(lre, nan=0.0, posinf=11.0), 0, 11)

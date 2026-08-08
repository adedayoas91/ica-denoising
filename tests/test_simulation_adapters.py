from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from csl.experiments import simulation_adapters as adapters


def test_estimate_cgc_prefers_fast_gcstar(monkeypatch):
    calls = []

    class FakeFastGcStar:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def fit(self, data):
            calls.append(("fit", data.shape))
            return self

        def get_result(
            self,
            *,
            alpha,
            beta,
            include_contemporaneous,
            use_fdr=False,
        ):
            calls.append(("get_result", alpha, beta, include_contemporaneous, use_fdr))
            scale = 2.0 if use_fdr else 1.0
            return SimpleNamespace(
                lag_only=scale
                * np.array(
                    [
                        [1.0, 0.2, 0.0],
                        [0.0, 1.0, 0.3],
                        [0.4, 0.0, 1.0],
                    ]
                ),
                warnings=("fast",),
            )

    def fail_if_slow_path_is_used():
        raise AssertionError("slow fit_cgc path should not be used")

    monkeypatch.setattr(adapters, "_load_fast_gcstar", lambda: FakeFastGcStar)
    monkeypatch.setattr(adapters, "_load_fit_cgc", fail_if_slow_path_is_used)

    estimate = adapters.estimate_cgc(
        np.ones((3, 25)),
        method="fcgc",
        n_perm=12,
        n_pasts=2,
        n_lags=2,
        alpha=0.05,
        beta=0.01,
        random_state=7,
    )

    assert estimate.estimator == "fcgc"
    assert estimate.warnings == ("fast",)
    assert estimate.binary.tolist() == [
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
    ]
    assert estimate.binary_fdr is None
    assert calls == [
        (
            "init",
            {
                "n_perm": 12,
                "n_pasts": 2,
                "n_lags": 2,
                "method": "fcgc",
                "random_state": 7,
            },
        ),
        ("fit", (3, 25)),
        ("get_result", 0.05, 0.01, False, False),
    ]


def test_estimate_cgc_computes_fdr_only_when_requested(monkeypatch):
    calls = []

    class FakeFastGcStar:
        def __init__(self, **_kwargs):
            pass

        def fit(self, data):
            calls.append(("fit", data.shape))
            return self

        def get_result(
            self,
            *,
            alpha,
            beta,
            include_contemporaneous,
            use_fdr=False,
        ):
            calls.append(("get_result", alpha, beta, include_contemporaneous, use_fdr))
            scale = 2.0 if use_fdr else 1.0
            return SimpleNamespace(
                lag_only=scale
                * np.array(
                    [
                        [1.0, 0.2],
                        [0.0, 1.0],
                    ]
                ),
                warnings=(),
            )

    monkeypatch.setattr(adapters, "_load_fast_gcstar", lambda: FakeFastGcStar)

    estimate = adapters.estimate_cgc(
        np.ones((2, 25)),
        compute_fdr=True,
        alpha=0.05,
        beta=0.01,
    )

    assert estimate.binary_fdr is not None
    assert estimate.binary_fdr.tolist() == estimate.binary.tolist()
    assert calls == [
        ("fit", (2, 25)),
        ("get_result", 0.05, 0.01, False, False),
        ("get_result", 0.05, 0.01, False, True),
    ]


def test_estimate_cgc_can_force_normal_backend(monkeypatch):
    calls = []

    def fail_if_fast_path_is_used():
        raise AssertionError("fast path should not be loaded")

    def fake_fit_cgc(data, **kwargs):
        calls.append((data.shape, kwargs))
        scale = 2.0 if kwargs.get("use_fdr") else 1.0
        return SimpleNamespace(
            lag_only=scale
            * np.array(
                [
                    [1.0, 0.2, 0.0],
                    [0.0, 1.0, 0.3],
                    [0.4, 0.0, 1.0],
                ]
            ),
            warnings=("normal",),
        )

    monkeypatch.setattr(adapters, "_load_fast_gcstar", fail_if_fast_path_is_used)
    monkeypatch.setattr(adapters, "_load_fit_cgc", lambda: fake_fit_cgc)

    estimate = adapters.estimate_cgc(
        np.ones((3, 25)),
        method="cgc",
        backend="normal",
        n_perm=12,
        n_pasts=2,
        n_lags=2,
        alpha=0.05,
        beta=0.01,
        random_state=7,
    )

    assert estimate.estimator == "cgc"
    assert estimate.warnings == ("normal",)
    assert estimate.binary.tolist() == [
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
    ]
    assert estimate.binary_fdr is None
    assert [kwargs.get("use_fdr", False) for _shape, kwargs in calls] == [False]
    assert calls[0][1] == {
        "n_perm": 12,
        "n_pasts": 2,
        "n_lags": 2,
        "method": "cgc",
        "random_state": 7,
        "alpha": 0.05,
        "beta": 0.01,
        "include_contemporaneous": False,
    }

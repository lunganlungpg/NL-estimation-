# Contributing

Thank you for helping make nonlinear estimation easier to learn and use. Clearer explanations are as welcome as code.

## Set up

```bash
git clone https://github.com/lunganlungpg/nonlinear-estimation.git
cd nonlinear-estimation
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
```

## Before opening a pull request

```bash
ruff check . && ruff format --check .
pytest
```

## Guidelines

- **Readable first.** `nlsfit` is a teaching tool. Prefer clear code with a comment that links it to the maths in `docs/theory.md`.
- **Verify against a reference.** New solvers or options must reproduce NIST certified values (LRE ≥ 6) or agree with SciPy.
- **New models** need an analytic Jacobian (it is tested against finite differences automatically) and, where possible, a `guess` function for starting values.
- **New datasets** go in `nlsfit/datasets.py` with their source and licence. Then run `python scripts/build_data.py` to regenerate `data/`.
- **Never report failure as success.** If a solver stops without reaching a minimum, `converged` must be False.
- **Website** changes go in `site/index.html` (no build step). Check light mode, dark mode and phone width.

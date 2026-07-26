# Web Dashboard

InferArena includes a lightweight [Streamlit](https://streamlit.io/) dashboard for
exploring experiment results without writing code.

## Install

```bash
pip install -e ".[dashboard]"
```

## Launch

```bash
inferarena dashboard
```

By default the dashboard reads from `./inferarena_outputs/`. Point it elsewhere
with:

```bash
inferarena dashboard --output-dir ./my_results
```

## Features

- **Summary table** — sortable metrics across all discovered experiments.
- **Latency CDF** — compare end-to-end latency distributions by scheduler.
- **Raw summaries** — inspect the underlying JSON for each experiment.

## When to use

Use the web dashboard for quick visual checks after running `inferarena compare`
or `inferarena benchmark`. For reproducible analysis and custom plots, use the
Jupyter notebook at `notebooks/dashboard.ipynb` or the `inferarena.reporting.dashboard`
Python helpers.

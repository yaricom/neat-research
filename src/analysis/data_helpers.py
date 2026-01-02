import pandas as pd

from analysis.model import ExperimentalData


def read_data_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    p_col, t_col, eff_col = infer_columns(df)

    cols = [p_col, t_col] + ([eff_col] if eff_col else [])
    d = df[cols].dropna().copy()
    d = d.rename(columns={p_col: "p", t_col: "T_ms", eff_col: "efficiency_score"})
    d["p"] = d["p"].astype(int)
    d = d.sort_values("p")

    return ExperimentalData(df=d, efficiency_score_column="efficiency_score", time_column="T_ms", p_column="p")


def infer_columns(df: pd.DataFrame):
    """Infer (p_col, time_col, eff_col) from CSV header."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # p/workers column
    p_col = None
    for c in df.columns:
        lc = c.lower()
        if lc in {"p", "workers", "worker", "num_workers", "n_workers", "goroutines"}:
            p_col = c
            break
    if p_col is None:
        # fallback: first column containing "work"
        for c in df.columns:
            if "work" in c.lower():
                p_col = c
                break
    if p_col is None:
        raise ValueError("Cannot infer workers column (p). Rename it to 'p' or include 'workers' in the name.")

    # time column: prefer ones containing "time"
    time_cols = [c for c in df.columns if "time" in c.lower()]
    if time_cols:
        # rank: prefer ms + total/epoch/generation
        def rank(c):
            lc = c.lower()
            r = 0
            if "ms" in lc or "msec" in lc:
                r += 3
            if "total" in lc or "epoch" in lc or "generation" in lc:
                r += 2
            if "avg" in lc or "mean" in lc:
                r += 1
            return -r
        t_col = sorted(time_cols, key=rank)[0]
    else:
        # fallback: first numeric column that isn't p and isn't efficiency
        t_col = None
        for c in df.columns:
            if c == p_col:
                continue
            if "efficiency" in c.lower():
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                t_col = c
                break
        if t_col is None:
            raise ValueError("Cannot infer time column. Add 'time' to the header, e.g. 'time_ms'.")

    # optional EfficiencyScore
    eff_col = None
    for c in df.columns:
        lc = c.lower()
        if "efficiencyscore" in lc or ("efficiency" in lc and "score" in lc):
            eff_col = c
            break

    return p_col, t_col, eff_col

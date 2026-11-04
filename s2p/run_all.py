"""Rebuild every table and figure from raw data:

    python -m s2p.run_all               # theory + train (tuning) + test (evaluation)
    python -m s2p.run_all --holdout     # ONE final run on the holdout with frozen choices
    python -m s2p.run_all --synthetic   # pipeline check on synthetic data (outputs marked SYNTHETIC)
"""
import argparse
import json
import pickle
import time

import numpy as np
import pandas as pd

from s2p import backtest, costs, research, signals, stats, synthetic, theory_report
from s2p.config import load_config
from s2p.data import load_panel
from s2p.plotstyle import INK2, MUTED, POLICY_COLOR, POLICY_LABEL, SERIES, plt, save

POLICIES = ["naive", "ema", "gp", "band"]


# ---------------------------------------------------------------- helpers

def md_table(df, digits=3):
    """Markdown table without extra dependencies."""
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:.{digits}f}"
        return str(v)
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def write(out, name, df, title, note=""):
    (out / f"{name}.md").write_text(f"## {title}\n\n{note}\n\n{md_table(df)}", encoding="utf-8")
    df.to_csv(out / f"{name}.csv", index=False)


def window(index, start, end):
    return (index >= pd.Timestamp(start)) & (index < pd.Timestamp(end) + pd.Timedelta(days=1))


def cached_panel(cfg, holdout, synthetic_data):
    if synthetic_data:
        return synthetic.make_panel(cfg)
    path = cfg["paths"]["cache"] / ("panel_holdout.pkl" if holdout else "panel.pkl")
    if path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)
    panel = load_panel(cfg, holdout=holdout)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(panel, fh)
    return panel


# ---------------------------------------------------------------- model context

def build_context(cfg, d, wins, selected, trials):
    """Everything the policies need. Train-only estimates are taken from `selected` if present
    (holdout run) and otherwise estimated on the train window and stored in `selected`."""
    names = list(cfg["signals"])
    index = d["vol"].index
    months = index.to_period("M").astype(str).to_numpy()
    X = research.stack(d["signals"], names)
    tradeable = d["tradeable"].to_numpy()
    y = research.ridge_target(d, cfg["model"]["target_clip"]).to_numpy()
    moments = research.monthly_moments(X, y, tradeable, months)
    order = list(pd.unique(months))
    first_fit = str(pd.Period(pd.Timestamp(cfg["splits"]["train"][0]), freq="M") + cfg["splits"]["burn_in_months"])
    train_score = wins["train"]

    if "alpha" not in selected:                         # ridge penalty: best OOS IC on train
        best = None
        for a in cfg["model"]["ridge_alphas"]:
            score, _, _ = research.predict(X, months, research.walk_forward(moments, order, first_fit, a))
            sc = pd.DataFrame(score, index=index, columns=d["vol"].columns).where(d["tradeable"])
            ic = research.ic_series(sc[train_score], d["fwd_ret"][train_score]).mean()
            trials.add("ridge_alpha", f"alpha={a}", "train", np.nan, mean_IC=ic)
            if best is None or ic > best[1]:
                best = (a, ic)
        selected["alpha"] = best[0]
    fits = research.walk_forward(moments, order, first_fit, selected["alpha"])
    score, beta, kappa = research.predict(X, months, fits)

    if "rho" not in selected:                           # signal decays from train only
        tr = window(index, *cfg["splits"]["train"])
        selected["rho"] = [research.ar1(d["signals"][n][tr]) for n in names]
        sc = pd.DataFrame(score, index=index).where(d["tradeable"].to_numpy())
        selected["rho_g"] = research.ar1(sc[train_score])
        # risk aversion: naive Markowitz has the target ex-ante vol on train
        ex_ante = np.nanmean(np.nansum(np.where(tradeable, score, np.nan)[train_score] ** 2, axis=1))
        per_bar = cfg["portfolio"]["target_vol_annual"] / np.sqrt(cfg["portfolio"]["bars_per_year"])
        selected["gamma"] = float(np.sqrt(ex_ante) / (per_bar * cfg["portfolio"]["aum_usd"]))

    rho = np.array(selected["rho"], float)
    if not (np.all(np.isfinite(rho)) and np.all((rho > 0) & (rho < 1)) and 0 < selected["rho_g"] < 1):
        raise ValueError(f"bad decay estimates {selected['rho']}, rho_g={selected['rho_g']}")
    c_lin, lam, q_ref = costs.cost_inputs(d, cfg)
    liquid = (costs.adv(d["quote_volume"], cfg["costs"]["adv_days"]) > 0).to_numpy()
    ok = tradeable & ~np.isnan(score) & np.isfinite(lam.to_numpy()) & liquid
    ctx = {"index": index, "columns": d["vol"].columns, "X": X, "beta": beta, "kappa": kappa,
           "score": np.nan_to_num(score), "vol": d["vol"].to_numpy(), "tradeable": ok,
           "fwd_ret": d["fwd_ret"].to_numpy(), "fwd_fund": d["fwd_funding"].to_numpy(),
           "c_lin": c_lin.to_numpy(), "lam": lam.to_numpy(), "q_ref": q_ref,
           "gamma": selected["gamma"], "rho": np.array(selected["rho"]), "discount": cfg["portfolio"]["discount"]}
    ctx["vol"] = np.where(ok, ctx["vol"], 1.0)          # avoid 0/0 where not tradeable (x forced to 0)
    ctx["kappa"] = np.nan_to_num(kappa, nan=1.0)
    ctx["band_tables"] = cached_band_tables(cfg, selected["rho_g"])
    ctx["fits"] = fits
    return ctx


def cached_band_tables(cfg, rho_g):
    path = cfg["paths"]["cache"] / f"band_tables_{rho_g:.6f}.pkl"
    if path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)
    t = backtest.band_tables(cfg, rho_g, cfg["portfolio"]["discount"], cfg["band_dp"]["c_grid"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(t, fh)
    return t


def scaled(ctx, cfg, d, **mult):
    """Copy of ctx with cost levels scaled (policies see and pay the scaled costs)."""
    c_lin, lam, _ = costs.cost_inputs(d, cfg, **mult)
    return {**ctx, "c_lin": c_lin.to_numpy(), "lam": lam.to_numpy()}


# ---------------------------------------------------------------- evaluation

def metrics(res, mask, cfg, rng):
    aum, bpy = cfg["portfolio"]["aum_usd"], cfg["portfolio"]["bars_per_year"]
    sc = cfg["stats"]
    r = res[mask]
    net = (r["net"] / aum).to_numpy()
    sr, lo, hi = stats.sharpe_ci(net, bpy, sc["block"], sc["n_boot"], rng)
    yrs = len(r) / bpy
    return {"net_SR": sr, "SR_lo95": lo, "SR_hi95": hi,
            "gross_SR": stats.sharpe((r["gross"] / aum).to_numpy(), bpy),
            "net_ret_pct": 100 * r["net"].sum() / aum / yrs,
            "gross_ret_pct": 100 * r["gross"].sum() / aum / yrs,
            "vol_pct": 100 * net.std(ddof=1) * np.sqrt(bpy),
            "cost_drag_pct": 100 * (r["cost_linear"] + r["cost_impact"]).sum() / aum / yrs,
            "impact_pct": 100 * r["cost_impact"].sum() / aum / yrs,
            "funding_pct": 100 * r["funding"].sum() / aum / yrs,
            "turnover_x_per_yr": r["traded"].sum() / aum / yrs,
            "gross_lev": r["gross_exposure"].mean() / aum,
            "n_bars": len(r), "n_eff": stats.effective_n(net), "n_blocks": len(r) // sc["block"]}


def tune(ctx, cfg, train, trials, rng):
    """Pick each policy's parameter by net Sharpe on the train window (1x costs)."""
    bpy, aum = cfg["portfolio"]["bars_per_year"], cfg["portfolio"]["aum_usd"]
    grids = {"naive": [None], "ema": cfg["policies"]["ema_rates"],
             "gp": cfg["policies"]["gp_cost_mults"], "band": cfg["policies"]["band_cost_mults"]}
    chosen, frontier = {}, []
    for kind, grid in grids.items():
        best = None
        for p in grid:
            res = backtest.run_policy(kind, p, ctx)
            r = res[train]
            sr = stats.sharpe((r["net"] / aum).to_numpy(), bpy)
            turn = r["traded"].sum() / aum / (len(r) / bpy)
            trials.add("policy", f"{kind}({p})", "train", sr, turnover=turn)
            frontier.append({"policy": kind, "param": p, "net_SR": sr, "turnover": turn})
            if best is None or sr > best[1]:
                best = (p, sr)
        chosen[kind] = best[0]
    return chosen, pd.DataFrame(frontier)


def edge_check(ctx, cfg, wins, trials):
    """Parameters beyond the pre-registered grid edge, scored on train (counted as trials)."""
    bpy, aum = cfg["portfolio"]["bars_per_year"], cfg["portfolio"]["aum_usd"]
    rows = []
    for kind, grid in cfg["policies"].get("edge_check", {}).items():
        for p in grid:
            res = backtest.run_policy(kind, p, ctx)
            sr = {w: stats.sharpe((res["net"][wins[w]] / aum).to_numpy(), bpy) for w in ["train", "test"]}
            trials.add("policy_edge_check", f"{kind}({p})", "train", sr["train"])
            rows.append({"policy": kind, "param": p, "train_net_SR": sr["train"],
                         "test_net_SR_posthoc": sr["test"],
                         "train_turnover": res["traded"][wins["train"]].sum() / aum / (wins["train"].sum() / bpy)})
    return pd.DataFrame(rows)


def evaluate(cfg, d, ctx, selected, wins, period, trials, rng, out):
    """Selected policies on one evaluation window: main table, paired differences, sensitivity."""
    aum, bpy, sc = cfg["portfolio"]["aum_usd"], cfg["portfolio"]["bars_per_year"], cfg["stats"]
    mask = wins[period]
    runs = {k: backtest.run_policy(k, selected["params"][k], ctx) for k in POLICIES}

    rows = []
    for k in POLICIES:
        m = metrics(runs[k], mask, cfg, rng)
        trials.add("evaluation", f"{k} 1x", period, m["net_SR"])
        rows.append({"policy": POLICY_LABEL[k], "param": selected["params"][k], **m})
    main = pd.DataFrame(rows)

    # paired block-bootstrap differences in net Sharpe vs naive and vs EMA
    net = {k: (runs[k]["net"][mask] / aum).to_numpy() for k in POLICIES}
    diff_rows = []
    for a, b in [("ema", "naive"), ("gp", "naive"), ("band", "naive"), ("gp", "ema"), ("band", "gp")]:
        f = lambda x, y: stats.sharpe(x, bpy) - stats.sharpe(y, bpy)
        draws = stats.bootstrap([net[a], net[b]], f, sc["block"], sc["n_boot"], rng)
        diff_rows.append({"comparison": f"{POLICY_LABEL[a]} - {POLICY_LABEL[b]}",
                          "delta_net_SR": f(net[a], net[b]),
                          "lo95": np.quantile(draws, 0.025), "hi95": np.quantile(draws, 0.975),
                          "share_draws_above_0": (draws > 0).mean()})
    diffs = pd.DataFrame(diff_rows)

    # cost sensitivity: all components scaled together, then one at a time
    sens = []
    scen = [(f"all x{m:g}", dict(fee=m, spread=m, impact=m)) for m in cfg["costs"]["multipliers"]]
    scen += [("fee x2", dict(fee=2)), ("spread x2", dict(spread=2)), ("impact x4", dict(impact=4))]
    for label, mult in scen:
        cx = scaled(ctx, cfg, d, **mult)
        for k in POLICIES:
            m = metrics(backtest.run_policy(k, selected["params"][k], cx), mask, cfg, rng)
            trials.add("evaluation", f"{k} {label}", period, m["net_SR"])
            sens.append({"costs": label, "policy": POLICY_LABEL[k], "net_SR": m["net_SR"],
                         "SR_lo95": m["SR_lo95"], "SR_hi95": m["SR_hi95"],
                         "net_ret_pct": m["net_ret_pct"], "turnover_x_per_yr": m["turnover_x_per_yr"]})
    sens = pd.DataFrame(sens)

    # signal-decay sensitivity: GP assumes half-lives x0.5 or x2 of the train estimate
    dec = []
    for dm in [1.0] + cfg["policies"]["decay_mults"]:
        m = metrics(backtest.run_policy("gp", selected["params"]["gp"], {**ctx, "decay_mult": dm}), mask, cfg, rng)
        trials.add("evaluation", f"gp decay x{dm}", period, m["net_SR"])
        dec.append({"assumed_half_life": f"x{dm:g}", "net_SR": m["net_SR"], "SR_lo95": m["SR_lo95"],
                    "SR_hi95": m["SR_hi95"], "turnover_x_per_yr": m["turnover_x_per_yr"],
                    "cost_drag_pct": m["cost_drag_pct"]})
    dec = pd.DataFrame(dec)
    return runs, main, diffs, sens, dec


def haircut(main, runs, wins, period, trials, cfg):
    """Selection-bias haircut: expected max Sharpe of N noise trials, and the deflated Sharpe."""
    bpy, aum = cfg["portfolio"]["bars_per_year"], cfg["portfolio"]["aum_usd"]
    n_trials = len(trials.frame())
    rows = []
    for k in POLICIES:
        net = (runs[k]["net"][wins[period]] / aum).to_numpy()
        # a zero-mean strategy's per-bar Sharpe estimate has standard error ~ 1 / sqrt(n_eff)
        sr0 = stats.expected_max_sharpe(n_trials, 1 / np.sqrt(stats.effective_n(net) - 1))
        rows.append({"policy": POLICY_LABEL[k], "net_SR": stats.sharpe(net, bpy),
                     "noise_max_SR": sr0 * np.sqrt(bpy), "n_trials": n_trials,
                     "prob_SR_above_0": stats.deflated_sharpe(net, 0.0),
                     "deflated_SR_prob": stats.deflated_sharpe(net, sr0)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures

def fig_ic(tables, path, synthetic_data):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    names = tables["train"]["signal"].tolist()
    yv = np.arange(len(names))
    for j, (period, color) in enumerate([("train", SERIES[0]), ("test", SERIES[1])]):
        if period not in tables:
            continue
        t = tables[period].set_index("signal").loc[names]
        se = t["mean_IC"] / t["IC_t_NW"]
        ax.errorbar(t["mean_IC"], yv + (0.15 if j else -0.15), xerr=1.96 * se, fmt="o", ms=6,
                    color=color, elinewidth=1.5, capsize=0, label=period)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(yv, names)
    ax.set_xlabel("mean cross-sectional IC with next-hour return (95% CI, Newey-West)")
    ax.set_title("Signal information coefficients")
    ax.legend(loc="lower right")
    save(fig, path, synthetic_data)


def fig_cum(runs, mask, aum, period, path, synthetic_data):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, kinds, title in [(axes[0], POLICIES, "(a) All four policies"),
                             (axes[1], POLICIES[1:], "(b) Without naive (zoom)")]:
        for k in kinds:
            c = 100 * runs[k]["net"][mask].cumsum() / aum
            ax.plot(c.index, c.to_numpy(), color=POLICY_COLOR[k], lw=1.6, label=POLICY_LABEL[k])
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_ylabel("cumulative net PnL, % of capital")
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8)
        ax.tick_params(axis="x", labelrotation=30)
    fig.suptitle(f"Net PnL after costs and funding ({period})", fontweight="bold")
    save(fig, path, synthetic_data)


def fig_cost_sens(sens, path, synthetic_data):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    s = sens[sens["costs"].str.startswith("all")].copy()
    s["m"] = s["costs"].str.replace("all x", "").astype(float)
    for ax, kinds, title in [(axes[0], POLICIES, "(a) All four policies"),
                             (axes[1], POLICIES[1:], "(b) Without naive (zoom)")]:
        for k in kinds:
            t = s[s.policy == POLICY_LABEL[k]]
            ax.plot(t["m"], t["net_SR"], "-o", ms=5, color=POLICY_COLOR[k], label=POLICY_LABEL[k])
            ax.fill_between(t["m"], t["SR_lo95"], t["SR_hi95"], color=POLICY_COLOR[k], alpha=0.10, lw=0)
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_xlabel("cost multiplier (fee, spread and impact together)")
        ax.set_ylabel("net Sharpe ratio (annualised)")
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle("Cost sensitivity (shaded: 95% block-bootstrap CI)", fontweight="bold")
    save(fig, path, synthetic_data)


def fig_frontier(frontier, path, synthetic_data):
    fig, ax = plt.subplots(figsize=(7, 4))
    for k in POLICIES:
        t = frontier[frontier.policy == k].sort_values("turnover")
        ax.plot(t["turnover"], t["net_SR"], "-o", ms=5, color=POLICY_COLOR[k], label=POLICY_LABEL[k])
    ax.set_xscale("log")
    ax.set_xlabel("turnover (x capital per year, log scale)")
    ax.set_ylabel("net Sharpe ratio, train")
    ax.set_title("Tuning grid on train: net Sharpe vs turnover")
    ax.legend()
    save(fig, path, synthetic_data)


# ---------------------------------------------------------------- main

def main(holdout=False, synthetic_data=False):
    t0 = time.time()
    cfg = load_config()
    out = cfg["paths"]["results"] / ("synthetic" if synthetic_data else "")
    out.mkdir(parents=True, exist_ok=True)
    tag = "SYNTHETIC DATA - pipeline check only. " if synthetic_data else ""
    rng = np.random.default_rng(cfg["seed"])
    sel_path = out / "selected.json"
    lock = out / "HOLDOUT_RUN.txt"
    if holdout:
        if lock.exists():
            raise SystemExit(f"holdout already run ({lock.read_text().strip()}); it may run only once")
        selected = json.loads(sel_path.read_text())        # frozen choices from the train run
    else:
        selected = {}
    trials = stats.TrialLog()
    if holdout:                                             # every earlier trial still counts
        trials.rows = pd.read_csv(out / "trials_test.csv").to_dict("records")

    if not holdout and not synthetic_data:
        print("theory layer ...")
        th = theory_report.run(cfg, out)
        for name, df in th.items():
            write(out, f"theory_{name}", df, f"Theory (simulated signals): {name}",
                  "Normalised single asset; utility as % of the frictionless optimum; every policy tuned.")

    print("loading panel ...")
    panel = cached_panel(cfg, holdout, synthetic_data)
    write(out, "data_quality_holdout" if holdout else "data_quality", panel["quality"], tag + "Data-quality summary")
    d = signals.build(panel, cfg)
    index = d["vol"].index
    first_fit = pd.Period(pd.Timestamp(cfg["splits"]["train"][0]), freq="M") + cfg["splits"]["burn_in_months"]
    wins = {"train": window(index, first_fit.start_time, cfg["splits"]["train"][1]),
            "test": window(index, *cfg["splits"]["test"])}
    if holdout:
        wins["holdout"] = window(index, *cfg["splits"]["holdout"])
    period = "holdout" if holdout else "test"

    print("signals ...")
    sig_tables = {}
    for p in ["train", period]:
        sig_tables[p] = research.signal_table(d["signals"], d, wins[p], cfg, rng)
        write(out, f"signals_{p}", sig_tables[p], tag + f"Single signals, {p} window",
              "IC = cross-sectional Spearman with next-hour return. p_holm = Holm-adjusted over the 6 signals.")
    for n in cfg["signals"]:
        trials.add("signal", n, "train", float(sig_tables["train"].set_index("signal").loc[n, "gross_SR"]))
    fig_ic(sig_tables if not holdout else {"train": sig_tables["train"], "test": sig_tables[period]},
           out / f"fig_ic_{period}.png", synthetic_data)

    print("model context ...")
    ctx = build_context(cfg, d, wins, selected, trials)
    score = pd.DataFrame(ctx["score"], index=index, columns=ctx["columns"]).where(ctx["tradeable"])
    comb = research.signal_table({"ridge_combined": score}, d, wins["train"], cfg, rng)
    comb = pd.concat([comb.assign(window="train"),
                      research.signal_table({"ridge_combined": score}, d, wins[period], cfg, rng).assign(window=period)])
    write(out, f"combined_{period}", comb, tag + "Ridge-combined signal (walk-forward, out of sample)")
    betas = pd.DataFrame([dict(month=m, kappa=k, **dict(zip(cfg["signals"], b))) for m, (b, k) in ctx["fits"].items()])
    write(out, "ridge_coefficients", betas, tag + "Walk-forward ridge coefficients (vol-scaled return per unit signal)")

    if "params" not in selected:
        print("tuning on train ...")
        selected["params"], frontier = tune(ctx, cfg, wins["train"], trials, rng)
        write(out, "tuning_train", frontier, tag + "Tuning grid on train (every row is a counted trial)")
        fig_frontier(frontier, out / "fig_frontier_train.png", synthetic_data)
        sel_path.write_text(json.dumps(selected, indent=1, default=float))
        edge = edge_check(ctx, cfg, wins, trials)
        write(out, "tuning_edge_check", edge, tag + "Grid-edge check (added after the first real run)",
              "Train decides; the test columns are POST-HOC and do not change any selection.")
    train_rows = pd.DataFrame([{"policy": POLICY_LABEL[k], **metrics(backtest.run_policy(k, selected["params"][k], ctx),
                                                                     wins["train"], cfg, rng)} for k in POLICIES])
    write(out, "policies_train", train_rows, tag + "Selected policies on train (in-sample for tuning)")

    print(f"evaluating on {period} ...")
    runs, main_t, diffs, sens, dec = evaluate(cfg, d, ctx, selected, wins, period, trials, rng, out)
    write(out, f"policies_{period}", main_t, tag + f"Selected policies, {period} window, 1x costs",
          "Returns are PnL / capital (USD 10m, no compounding). SR CI: circular block bootstrap, 1-week blocks.")
    write(out, f"differences_{period}", diffs, tag + f"Paired differences in net Sharpe, {period}")
    write(out, f"cost_sensitivity_{period}", sens, tag + f"Cost sensitivity, {period}")
    write(out, f"decay_sensitivity_{period}", dec, tag + f"GP with mis-specified signal decay, {period}")
    write(out, f"haircut_{period}", haircut(main_t, runs, wins, period, trials, cfg),
          tag + f"Selection-bias haircut, {period}",
          "noise_max_SR = expected max Sharpe of n_trials pure-noise strategies; deflated_SR_prob = P(true SR > that).")
    fig_cum(runs, wins[period], cfg["portfolio"]["aum_usd"], period, out / f"fig_pnl_{period}.png", synthetic_data)
    fig_cost_sens(sens, out / f"fig_cost_sensitivity_{period}.png", synthetic_data)
    trials.frame().to_csv(out / f"trials_{period}.csv", index=False)
    json.dump({"selected": selected, "n_trials": len(trials.frame()), "seconds": time.time() - t0},
              open(out / f"run_{period}.json", "w"), indent=1, default=float)
    if holdout:
        lock.write_text(f"run at {pd.Timestamp.now(tz='UTC')}")
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true", help="final one-off run on the holdout")
    ap.add_argument("--synthetic", action="store_true", help="pipeline check on synthetic data")
    a = ap.parse_args()
    main(holdout=a.holdout, synthetic_data=a.synthetic)

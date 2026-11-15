"""
LightGBM models + the safer gambling guardrail, judged against the baselines.

  1. Churn   LightGBM classifier  vs  baseline best (days since last bet)
  2. Value   LightGBM regressor   vs  BG/NBD baseline
  3. Harm    rules-only guardrail  vs  LightGBM model + rules
             (the model threshold is chosen on TRAINING data only)
  4. Offers  top 10% by predicted value, with and without the guardrail
  5. SHAP    why the harm model flags players, in plain terms

Uses the same players and the same 70/30 split as train_baselines.py, so
every comparison is like-for-like.

Usage:
    python -m pve.train_models
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from pve import guardrail as g
from pve.train_baselines import CATEGORICAL, NOT_FEATURES, load, top_decile_capture

REPORTS = Path("reports")
SEED = 42
LGB_PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31,
                  min_child_samples=30, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, random_state=SEED, verbose=-1)


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """All snapshot features. Labels and hidden '_' columns are excluded."""
    drop = NOT_FEATURES + ["churned_90d"] + [c for c in df.columns
                                              if c.startswith("future_") or c.startswith("_")]
    X = df.drop(columns=drop).copy()
    for c in CATEGORICAL:                    # LightGBM handles categories natively
        X[c] = X[c].astype("category")
    return X


def main() -> None:
    df = load()
    train, test = train_test_split(df, test_size=0.3, random_state=SEED, stratify=df["churned_90d"])
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)
    X_tr, X_te = feature_matrix(train), feature_matrix(test)
    baseline = json.loads((REPORTS / "baseline_metrics.json").read_text())
    report: dict = {"players": {"train": len(train), "test": len(test)}}

    # ------------------------------------------------------------ 1. churn
    churn = lgb.LGBMClassifier(**LGB_PARAMS).fit(X_tr, train["churned_90d"].astype(int))
    p_churn = churn.predict_proba(X_te)[:, 1]
    y_churn = test["churned_90d"].astype(int)
    report["churn"] = {
        "roc_auc_lightgbm": round(float(roc_auc_score(y_churn, p_churn)), 3),
        "pr_auc_lightgbm": round(float(average_precision_score(y_churn, p_churn)), 3),
        "roc_auc_best_baseline": max(baseline["churn"]["roc_auc"].values()),
    }

    # ------------------------------------------------------------ 2. value
    # predict future STAKE (always >= 0, far less luck-driven than net loss),
    # on a log scale because a few players stake enormous amounts
    value = lgb.LGBMRegressor(**LGB_PARAMS).fit(X_tr, np.log1p(train["future_stake"]))
    margin = train["total_net_loss"].sum() / train["total_stake"].sum()
    pred_clv = np.expm1(value.predict(X_te)) * margin
    actual = test["future_net_loss"].to_numpy()
    report["value"] = {
        "spearman_lightgbm": round(float(spearmanr(pred_clv, actual)[0]), 3),
        "top_decile_capture_lightgbm": round(top_decile_capture(pred_clv, actual), 3),
        "spearman_baseline": baseline["clv"]["value"]["spearman_clv_model"],
        "top_decile_capture_baseline": baseline["clv"]["value"]["top_decile_capture_clv_model"],
    }

    # ------------------------------------------------------------ 3. harm
    # Label: players already in harm at the snapshot. In real life this would come
    # from past safer gambling interactions and self-exclusions.
    y_tr, y_te = train["_in_harm"].astype(int), test["_in_harm"].astype(int)
    harm = lgb.LGBMClassifier(**LGB_PARAMS)
    # thresholds are chosen on out-of-fold TRAINING predictions, never on the test set
    oof = cross_val_predict(harm, X_tr, y_tr, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                            method="predict_proba")[:, 1]
    harm.fit(X_tr, y_tr)
    p_harm = pd.Series(harm.predict_proba(X_te)[:, 1])

    rules_te = g.rules_score(test)
    model_amber_at = g.threshold_for_recall(oof, y_tr, g.TARGET_RECALL)
    model_flag = p_harm >= model_amber_at
    guardrails = {
        # (score used for AUC, amber mask, red mask)
        "rules_only": (rules_te, rules_te >= g.RULES_AMBER, rules_te >= g.RULES_RED),
        "model_plus_rules": (p_harm, model_flag, model_flag & (rules_te >= g.RED_RULE_POINTS)),
    }
    seg = test["_true_segment"].where(~test["_in_harm"], "at_risk_in_harm")
    harm_report = {"harmed_players_in_test": int(y_te.sum()),
                   "model_amber_threshold": round(float(model_amber_at), 4)}
    decisions = {}
    for name, (score, amber, red) in guardrails.items():
        d = g.decide(test["sof_status"], amber, red)
        decisions[name] = d
        flagged = d.isin(["amber", "red", "suspended"])
        days_since_onset = (test["snapshot_date"] - test["_harm_onset_date"]).dt.days
        harm_report[name] = {
            "roc_auc": round(float(roc_auc_score(y_te, score)), 3),
            "harmed_caught": round(float(flagged[y_te == 1].mean()), 3),
            "harmed_caught_red": round(float((d == "red")[y_te == 1].mean()), 3),
            "precision_of_flags": round(float(y_te[flagged].mean()), 3),
            "red_precision": round(float(y_te[d == "red"].mean()), 3) if (d == "red").any() else None,
            "players_flagged": int(flagged.sum()),
            "share_flagged_by_group": {k: round(float(v), 3)
                                       for k, v in flagged.groupby(seg).mean().items()},
            "median_days_since_onset_caught": float(days_since_onset[flagged & (y_te == 1)].median()),
            "median_days_since_onset_missed": float(days_since_onset[~flagged & (y_te == 1)].median()),
        }
    report["harm"] = harm_report

    # ------------------------------------------------------------ 4. offers
    # Retention budget: offers for 10% of current players, chosen by predicted value.
    n_offers = len(test) // 10
    offers = pd.DataFrame({"pred_clv": pred_clv, "in_harm": y_te.astype(bool),
                           "actual_value": actual, "decision": decisions["model_plus_rules"]})
    no_guard = offers.nlargest(n_offers, "pred_clv")
    with_guard = offers[offers["decision"] == "eligible"].nlargest(n_offers, "pred_clv")
    report["offers"] = {
        "offers_sent": n_offers,
        "harmed_share_without_guardrail": round(float(no_guard["in_harm"].mean()), 3),
        "harmed_share_with_guardrail": round(float(with_guard["in_harm"].mean()), 3),
        "harmed_players_offered_without_guardrail": int(no_guard["in_harm"].sum()),
        "harmed_players_offered_with_guardrail": int(with_guard["in_harm"].sum()),
        "future_value_targeted_without_guardrail": round(float(no_guard["actual_value"].sum())),
        "future_value_targeted_with_guardrail": round(float(with_guard["actual_value"].sum())),
    }

    # ------------------------------------------------------------ 5. SHAP
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sv = shap.TreeExplainer(harm).shap_values(X_te)
    sv = sv[1] if isinstance(sv, list) else sv
    importance = pd.Series(np.abs(sv).mean(axis=0), index=X_te.columns).sort_values(ascending=False)
    report["harm_top_features"] = {k: round(float(v), 4) for k, v in importance.head(10).items()}
    examples = []
    red_idx = np.where((decisions["model_plus_rules"] == "red").to_numpy())[0][:3]
    for i in red_idx:
        top = pd.Series(sv[i], index=X_te.columns).sort_values(ascending=False).head(3)
        examples.append({
            "player_id": int(test.loc[i, "player_id"]),
            "p_harm": round(float(p_harm[i]), 3),
            "top_reasons": [f"{f} = {X_te.loc[i, f]}" if not isinstance(X_te.loc[i, f], float)
                            else f"{f} = {X_te.loc[i, f]:.2f}" for f in top.index],
            "retail_rules_triggered": g.rule_reasons(test.loc[i]),
        })
    report["red_flag_examples"] = examples

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "model_metrics.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()

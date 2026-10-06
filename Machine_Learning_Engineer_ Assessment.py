import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

DATA = "/content/DATA/"
TRAIN_FILE = DATA + "train-test.csv"
VALID_FILE = DATA + "validation.csv"
TEMPLATE_FILE = DATA + "validation-predictions-template.csv"
DEC_FILE = DATA + "december-chart-inputs.csv"

TARGET, ID, DATE = "posted_rate", "load_id", "date"
PICKUP, DELIVERY = "pickup", "delivery"
CATS = [PICKUP, DELIVERY, "equipment", "lane"]
SIGNALS = ["market_index", "quote_signal"]

def load(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df[DATE] = pd.to_datetime(df[DATE])
    return df

def clean(df, is_train, weight_med=None):
    df = df.copy()
    df["equipment"] = df["equipment"].astype(str).str.strip().str.title()
    for c in (PICKUP, DELIVERY):
        df[c] = df[c].astype(str).str.strip().str.title().replace({"Nan": np.nan})
    if is_train:
        df = df.dropna(subset=[TARGET, DATE])
        df = df[(df[TARGET] > 0) & (df["distance"] > 0)]
        rpm = df[TARGET] / df["distance"]
        lo, hi = rpm.quantile([0.005, 0.995])
        df = df[(rpm >= lo) & (rpm <= hi)]
    df["weight"] = df["weight"].fillna(weight_med if weight_med is not None else df["weight"].median())
    return df

def features(df):
    df = df.copy()
    d = df[DATE]
    df["lane"] = df[PICKUP].astype(str) + "->" + df[DELIVERY].astype(str)
    df["dow"] = d.dt.dayofweek
    df["month"] = d.dt.month
    df["dom"] = d.dt.day
    df["woy"] = d.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    x_this = pd.to_datetime(d.dt.year.astype(str) + "-12-25")
    x_prev = pd.to_datetime((d.dt.year - 1).astype(str) + "-12-25")
    d_this, d_prev = (x_this - d).dt.days, (x_prev - d).dt.days
    df["days_to_xmas"] = np.where(d_this.abs() <= d_prev.abs(), d_this, d_prev)
    df["holiday_window"] = (df["days_to_xmas"].abs() <= 7).astype(int)
    df["log_dist"] = np.log1p(df["distance"])
    df["weight_k"] = df["weight"] / 1000
    if "quote_signal" in df:
        df["implied_rate"] = df["quote_signal"] * df["distance"]
        df["implied_x_mkt"] = df["implied_rate"] * df["market_index"]
    return df

def set_cats(frames):
    allv = pd.concat(frames)
    for c in CATS:
        cat = pd.CategoricalDtype(sorted(allv[c].astype(str).unique()))
        for f in frames:
            f[c] = f[c].astype(str).astype(cat)

def feature_cols(df):
    return [c for c in df.columns if c not in (TARGET, ID, DATE)]


PARAMS = dict(objective="regression_l1", learning_rate=0.08, n_estimators=3000,
              num_leaves=63, min_child_samples=20, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, cat_smooth=20, verbose=-1, n_jobs=-1)

def fit(train, valid=None, n_estimators=None, cols=None):
    cols = cols or feature_cols(train)
    p = dict(PARAMS)
    if n_estimators:
        p["n_estimators"] = n_estimators
    m = lgb.LGBMRegressor(**p)
    kw = {}
    if valid is not None:
        kw = dict(eval_set=[(valid[cols], np.log1p(valid[TARGET]))],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    m.fit(train[cols], np.log1p(train[TARGET]), **kw)
    return m, cols

def evaluate(m, cols, df):
    pred = np.expm1(m.predict(df[cols]))
    return mean_absolute_error(df[TARGET], pred), mean_absolute_percentage_error(df[TARGET], pred)

def diagnostics(tr, va):
    print("train dates:", tr[DATE].min().date(), "->", tr[DATE].max().date())
    print("valid dates:", va[DATE].min().date(), "->", va[DATE].max().date())
    print(f"% valid rows inside train date range: {(va[DATE] <= tr[DATE].max()).mean():.1%}")
    for c in [PICKUP, DELIVERY, "equipment", "lane"]:
        print(f"unseen {c} in validation: {(~va[c].isin(set(tr[c]))).mean():.1%}")
    print("columns missing in validation:", [c for c in tr.columns if c not in va.columns])

def main():

    raw_train = load(TRAIN_FILE)
    w_med = raw_train["weight"].median()
    tr = features(clean(raw_train, True, w_med))
    va = features(clean(load(VALID_FILE), False, w_med))


    dec_raw = load(DEC_FILE)
    dec_raw = dec_raw.rename(columns={dec_raw.columns[0]: PICKUP})
    dec_raw = clean(dec_raw, False, w_med)
    dec_raw[PICKUP] = dec_raw[PICKUP].fillna("Lexington")

    last = tr.sort_values(DATE).tail(2000)
    va_daily = va.groupby(DATE)[SIGNALS].median()
    for c in SIGNALS:
        if c not in dec_raw:
            dec_raw[c] = dec_raw[DATE].map(va_daily[c])
            dec_raw[c] = dec_raw[c].fillna(last[c].median())
    geo_p = tr.groupby(PICKUP)[["pickup_lat", "pickup_lon"]].median()
    geo_d = tr.groupby(DELIVERY)[["delivery_lat", "delivery_lon"]].median()
    dec_raw = dec_raw.join(geo_p, on=PICKUP).join(geo_d, on=DELIVERY)
    dec = features(dec_raw)

    diagnostics(tr, va)
    set_cats([tr, va, dec])


    tr = tr.sort_values(DATE).reset_index(drop=True)
    cols = [c for c in feature_cols(tr) if c in va.columns]
    cut = tr[DATE].quantile(0.85)
    fit_df, hold = tr[tr[DATE] <= cut], tr[tr[DATE] > cut]
    m, cols = fit(fit_df, hold, cols=cols)
    mae, mape = evaluate(m, cols, hold)
    print(f"Holdout (latest 15%)  MAE={mae:.2f}  MAPE={mape:.3%}  best_iter={m.best_iteration_}")


    seen = hold["lane"].isin(set(fit_df["lane"]))
    for name, mask in [("seen lanes", seen), ("unseen lanes", ~seen)]:
        if mask.any():
            print(f"  {name}: n={mask.sum()}  MAE={evaluate(m, cols, hold[mask])[0]:.2f}")
    base = hold["quote_signal"] * hold["distance"]
    print(f"  baseline (quote_signal x distance) MAE={mean_absolute_error(hold[TARGET], base):.2f}")

    qs = [0.5, 0.65, 0.8]
    for a, b in zip(qs, qs[1:] + [1.0]):
        t0, t1 = tr[DATE].quantile(a), tr[DATE].quantile(b)
        f, h = tr[tr[DATE] <= t0], tr[(tr[DATE] > t0) & (tr[DATE] <= t1)]
        mm, cc = fit(f, h, cols=cols)
        print(f"  fold up to {t0.date()}: MAE={evaluate(mm, cc, h)[0]:.2f}")


    final, cols = fit(tr, n_estimators=int(m.best_iteration_ * 1.1), cols=cols)


    out = pd.read_csv(TEMPLATE_FILE)
    pred = pd.Series(np.expm1(final.predict(va[cols])), index=va[ID].values)
    out["predicted_rate"] = out[ID].map(pred).round(2)
    assert out["predicted_rate"].notna().all() and len(out) == 12000
    out[[ID, "predicted_rate"]].to_csv("validation_predictions.csv", index=False)


    dec_out = pd.read_csv(DEC_FILE)
    dec_out.iloc[:, 0] = dec_out.iloc[:, 0].fillna("Lexington")
    dec_out["predicted_rate"] = np.expm1(final.predict(dec[cols])).round(2)
    dec_out.to_csv("december_chart_inputs_filled.csv", index=False)
    print("Done. Now run score.py per README.")


if __name__ == "__main__":
    main()

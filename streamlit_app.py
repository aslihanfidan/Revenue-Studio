# Streamlit Revenue Studio — Movie Revenue Prediction (Adapted to your previous file)
# ---------------------------------------------------------------------------------
# This version preserves your original layout (EDA / Öneri / Modelleme / Tahmin)
# and adds hooks to your **Project_final_2.py** pipeline:
#   - prepare_and_train() for advanced FE + TF‑IDF+SVD + LightGBM
#   - run_stacking() for LGBM+XGB+CatBoost OOF stacking (meta: RidgeCV)
#   - build_recommender()/recommend_* for SBERT-based recommendations
#
# You can run a quick baseline (Linear/RF/LGBM/XGB) OR the advanced pipeline.
# Test CSV upload is tucked away (default: only train.csv) per your team's feedback.
#
# How to run
#   1) Put this file as `streamlit_app.py` in a folder together with `Project_final_2.py`.
#   2) (Recommended) Create & activate a virtual env.
#   3) Install requirements (pick what you need):
#        pip install -U streamlit pandas numpy scikit-learn plotly matplotlib joblib
#        pip install lightgbm xgboost catboost sentence-transformers shap
#   4) Start:
#        streamlit run streamlit_app.py
# ---------------------------------------------------------------------------------

from __future__ import annotations
import io
import os
import ast
import typing as T

import numpy as np
import pandas as pd

import streamlit as st

from inspect import signature
from datetime import datetime

# Sklearn
from sklearn.model_selection import KFold, train_test_split, learning_curve
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Optional models
try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# Plotting
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# SHAP (optional)
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# Import your project pipeline (soft import)
PROJ = None
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("Project_final_2", os.path.join(os.getcwd(), "Project_final_2.py"))
    if spec and spec.loader:
        PROJ = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(PROJ)
except Exception as _e:
    PROJ = None

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="Revenue Studio — Movie Revenue Prediction",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Utility: version-safe RMSE
# -----------------------------
def rmse_score(y_true, y_pred) -> float:
    """Version-safe RMSE calculation for old/new scikit-learn."""
    try:
        if 'squared' in signature(mean_squared_error).parameters:
            return float(mean_squared_error(y_true, y_pred, squared=False))
    except Exception:
        pass
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

# -----------------------------
# Sidebar — Data Input
# -----------------------------
st.sidebar.header("📥 Veri Girişi")
use_only_train = st.sidebar.checkbox("Sadece train.csv ile çalış", value=True)

st.sidebar.markdown("**Beklenen hedef sütunu:** `revenue` (sayısal)")
train_file = st.sidebar.file_uploader("train.csv (hedef: revenue)", type=["csv"], key="train")

test_file = None
with st.sidebar.expander("(Opsiyonel) test.csv yükle"):
    if not use_only_train:
        test_file = st.file_uploader("test.csv (hedefsiz)", type=["csv"], key="test")
    else:
        st.caption("Feedback doğrultusunda varsayılan: test kullanılmıyor.")

split_data = st.sidebar.checkbox("Tek CSV yüklersem eğitim/teste böl", value=False, disabled=use_only_train is False)
split_ratio = st.sidebar.slider("Test oranı", 0.1, 0.5, 0.2, 0.05, disabled=(not split_data))
random_state = st.sidebar.number_input("Random state", min_value=0, value=42, step=1)

# -----------------------------
# Data Load Helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def _read_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

@st.cache_data(show_spinner=True)
def load_datasets(train_f, test_f, do_split, split_ratio, seed, only_train) -> T.Tuple[pd.DataFrame, pd.DataFrame | None]:
    if train_f is not None and (not do_split) and only_train:
        train_df = _read_csv(train_f)
        return train_df, None

    if train_f is not None and (not only_train) and (not do_split):
        train_df = _read_csv(train_f)
        test_df = _read_csv(test_f) if test_f is not None else None
        return train_df, test_df

    if train_f is not None and do_split:
        full = _read_csv(train_f)
        assert "revenue" in full.columns, "Veride 'revenue' hedef sütunu bulunmalı."
        train_df, valid_df = train_test_split(full, test_size=split_ratio, random_state=seed)
        valid_df = valid_df.drop(columns=["revenue"])  # hedefsiz test gibi
        return train_df.reset_index(drop=True), valid_df.reset_index(drop=True)

    return pd.DataFrame(), None

train_df_raw, test_df_raw = load_datasets(train_file, test_file, split_data, split_ratio, random_state, use_only_train)

# -----------------------------
# Main Title
# -----------------------------
st.title("💸 Streamlit Revenue Studio")
st.write("Film gelir tahmini için uçtan uca bir ML uygulaması. Yükle, eğit, karşılaştır, incele ve indir.")

# -----------------------------
# Tabs
# -----------------------------
eda_tab, reco_tab, model_tab, predict_tab = st.tabs(
    ["🔎 EDA / Ön İzleme a", "🎬 Öneri", "🤖 Modelleme", "🎯 Tahmin Playground"]
)

# -----------------------------
# EDA Panel
# -----------------------------
with eda_tab:
    st.subheader("Veri Önizleme")
    if train_df_raw.empty:
        st.info("Sol taraftan train dosyasını yükleyin. (İsterseniz tek CSV'yi eğitim/teste bölebilirsiniz)")
    else:
        n_show = st.slider("Gösterilecek satır sayısı", 5, 50, 10)
        st.write("**Train örnekleri:**")
        st.dataframe(train_df_raw.head(n_show))
        if test_df_raw is not None:
            st.write("**Test örnekleri:**")
            st.dataframe(test_df_raw.head(n_show))

        with st.expander("Eksik Değerler (Train)"):
            miss = train_df_raw.isna().sum().sort_values(ascending=False)
            st.dataframe(miss.to_frame("missing"))
        if "revenue" in train_df_raw.columns:
            st.success(f"Hedef sütun bulundu: revenue ✅  (n={train_df_raw['revenue'].notna().sum()})")
        else:
            st.warning("Hedef sütun bulunamadı: `revenue` bekleniyor.")

        # Basit sayısal dağılım
        num_cols = train_df_raw.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            col_x = st.selectbox("Sayısal dağılım grafiği için sütun seç", num_cols, index=0)
            fig = px.histogram(train_df_raw, x=col_x, nbins=50, title=f"Dağılım — {col_x}")
            st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Recommendation Tab
# -----------------------------
with reco_tab:
    st.subheader("Benzer Film Önerisi")

    if train_df_raw.empty:
        st.info("Önce sol menüden train.csv yükleyin. (Tercihen: title / overview / genres sütunları)")
    else:
        mode_rec = st.radio("Öneri modu", ["Basit TF‑IDF", "Gelişmiş SBERT (Project_final_2)"] , index=0, horizontal=True)

        # --- Helpers ---
        def _pick_col(df, candidates):
            for c in candidates:
                if c in df.columns:
                    return c
            low = {c.lower(): c for c in df.columns}
            for c in candidates:
                if c.lower() in low:
                    return low[c.lower()]
            return None

        if mode_rec == "Basit TF‑IDF":
            def _combined_text(df, title_col, overview_col, genre_col):
                parts = []
                if overview_col: parts.append(df[overview_col].astype(str))
                if genre_col:    parts.append(df[genre_col].astype(str))
                if not parts:    return pd.Series([""] * len(df))
                s = pd.Series([""] * len(df))
                for p in parts: s = s.str.cat(p.fillna(""), sep=" ")
                return s.str.replace(r"\s+", " ", regex=True).str.strip()

            title_col    = _pick_col(train_df_raw, ["title","original_title","movie_title","Title"])
            overview_col = _pick_col(train_df_raw, ["overview","description","plot","tagline"])
            genre_col    = _pick_col(train_df_raw, ["genres","genre","main_genre"])

            if not title_col:
                st.error("Film başlığı sütunu bulunamadı (title / original_title vb.).")
            elif not (overview_col or genre_col):
                st.error("overview/description veya genres/genre sütunlarından en az biri gerekli.")
            else:
                titles = (
                    train_df_raw[title_col]
                    .astype(str).dropna().drop_duplicates().sort_values()
                    .tolist()
                )
                film_sec = st.selectbox("Film seç:", titles, index=0)
                if st.button("🔎 Benzer 5 filmi getir", type="primary"):
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    from sklearn.metrics.pairwise import cosine_similarity

                    df = train_df_raw[[title_col] + [c for c in [overview_col, genre_col] if c]].copy()
                    text = _combined_text(df, title_col, overview_col, genre_col)
                    vec = TfidfVectorizer(stop_words="english", max_features=50000, ngram_range=(1,2))
                    X = vec.fit_transform(text.fillna(""))

                    # index of selected title (case-insensitive)
                    try:
                        idx = df[title_col].astype(str).str.lower().tolist().index(film_sec.lower())
                    except ValueError:
                        matches = [i for i, t in enumerate(df[title_col].astype(str)) if t.lower()==film_sec.lower()]
                        idx = matches[0] if matches else 0

                    sims = cosine_similarity(X[idx], X).ravel()
                    sims[idx] = -1.0
                    top_idx = sims.argsort()[::-1][:5]

                    recs = df.iloc[top_idx].copy()
                    recs["similarity"] = sims[top_idx]
                    meta_cols = [c for c in ["vote_average","popularity","budget","revenue"] if c in train_df_raw.columns]
                    for c in meta_cols:
                        recs[c] = train_df_raw.iloc[top_idx][c].values

                    st.write(f"**Seçilen film:** {film_sec}")
                    st.dataframe(
                        recs[[title_col] + ([genre_col] if genre_col else []) + meta_cols + ["similarity"]],
                        use_container_width=True
                    )

        else:
            if PROJ is None:
                st.error("Project_final_2.py bulunamadı ya da içe aktarılamadı. Dosyayı aynı klasöre koyun.")
            else:
                with st.spinner("SBERT gömmeleri hazırlanıyor (ilk çalıştırmada biraz sürebilir)..."):
                    @st.cache_resource
                    def _build_rec(df_all):
                        rec = PROJ.build_recommender(df_all)
                        return PROJ.patch_title_index(rec)
                    rec = _build_rec(train_df_raw)

                titles = sorted(pd.Series(rec["titles_full"]).dropna().astype(str).unique().tolist())
                film_sec = st.selectbox("Film seç:", titles, index=0)
                k = st.slider("Öneri sayısı", 3, 10, 5)
                t1, t2, t3 = st.tabs(["Non‑franchise", "Hybrid", "Serbest arama"])
                with t1:
                    st.dataframe(PROJ.recommend_by_title_nonfranchise(film_sec, rec, top_k=k))
                with t2:
                    st.dataframe(PROJ.recommend_hybrid(film_sec, rec, top_k=k))
                with t3:
                    q = st.text_input("Film/vibe tarif et:", film_sec)
                    if q:
                        st.dataframe(PROJ.recommend_by_query(q, rec, top_k=k))

# -----------------------------
# Modeling Tab
# -----------------------------
with model_tab:
    st.subheader("Model Eğitimi ve Karşılaştırma")

    if train_df_raw.empty:
        st.info("Önce veri yükleyin.")
    else:
        # ===== Quick Baseline (your original flow) =====
        st.markdown("#### 1) Hızlı Karşılaştırma (basit önişleme)")
        do_log_transform = st.checkbox("Gelir (revenue) için log1p dönüşümü uygula", value=True)
        n_splits = st.slider("K‑Fold (CV) kat sayısı", 3, 10, 5, key="cv_quick")

        models_available = {
            "Linear Regression": True,
            "RandomForest": True,
            "LightGBM": HAS_LGBM,
            "XGBoost": HAS_XGB,
        }
        chosen = [m for m, ok in models_available.items() if ok and st.checkbox(m, value=(m in ["RandomForest","LightGBM"]), key=f"chk_{m}")]
        if not chosen:
            st.warning("En az bir model seçin.")

        with st.expander("RandomForest Hiperparametreleri"):
            rf_n_estimators = st.slider("n_estimators", 50, 500, 200, 50)
            rf_max_depth = st.selectbox("max_depth", options=[None, 5, 10, 15, 20], index=2)
            rf_min_samples_split = st.selectbox("min_samples_split", options=[2, 5, 10], index=0)

        # Minimal preprocessing aligned to quick baseline
        @st.cache_data(show_spinner=True)
        def preprocess_quick(train_df: pd.DataFrame, test_df: pd.DataFrame | None) -> T.Tuple[pd.DataFrame, pd.DataFrame | None, pd.Series]:
            df_train = train_df.copy()
            df_test = test_df.copy() if test_df is not None else None
            assert 'revenue' in df_train.columns, "Train verisinde 'revenue' olmalı."

            # Zero→NaN then fill
            for c in ["budget", "runtime"]:
                if c in df_train.columns:
                    df_train[c] = df_train[c].replace(0, np.nan)
            if df_test is not None:
                for c in ["budget", "runtime"]:
                    if c in df_test.columns:
                        df_test[c] = df_test[c].replace(0, np.nan)

            for c in ["budget", "runtime"]:
                if c in df_train.columns:
                    med = df_train[c].median()
                    df_train[c] = df_train[c].fillna(med)
                    if df_test is not None and c in df_test.columns:
                        df_test[c] = df_test[c].fillna(med)

            # release_date features
            def _date_feats(d):
                d['release_date'] = pd.to_datetime(d['release_date'], errors='coerce')
                d['release_year'] = d['release_date'].dt.year
                d['release_month'] = d['release_date'].dt.month
                d['release_dayofweek'] = d['release_date'].dt.dayofweek
                return d
            if 'release_date' in df_train.columns:
                df_train = _date_feats(df_train)
            if df_test is not None and 'release_date' in df_test.columns:
                df_test = _date_feats(df_test)

            # Simple categorical dummies
            full = pd.concat([df_train.drop(columns=['revenue']), df_test], axis=0, ignore_index=True) if df_test is not None else df_train.drop(columns=['revenue'])
            obj_cols = full.select_dtypes(include=['object']).columns.tolist()
            if obj_cols:
                dummies = pd.get_dummies(full[obj_cols], dummy_na=False)
                full = pd.concat([full.drop(columns=obj_cols), dummies], axis=1)

            if df_test is not None:
                X_train = full.iloc[:len(df_train)].reset_index(drop=True)
                X_test = full.iloc[len(df_train):].reset_index(drop=True)
            else:
                X_train = full.reset_index(drop=True)
                X_test = None

            y = df_train['revenue'].copy()
            return X_train, X_test, y

        X_train_q, X_test_q, y_q = preprocess_quick(train_df_raw, test_df_raw)
        y_train_target_q = np.log1p(y_q) if do_log_transform else y_q

        st.write(f"**Eğitim boyutu:** {X_train_q.shape} | **Hedef:** {y_train_target_q.shape}")

        if st.button("🚀 Eğit & Karşılaştır (Hızlı)", type="primary"):
            with st.spinner("Modeller eğitiliyor ve çapraz doğrulama yapılıyor..."):
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

                results = []
                feature_importances: dict[str, np.ndarray] = {}

                for model_name in chosen:
                    fold_rmse_log, fold_rmse_dollar, fold_r2 = [], [], []

                    # Build model
                    if model_name == "Linear Regression":
                        model = LinearRegression()
                    elif model_name == "RandomForest":
                        model = RandomForestRegressor(
                            n_estimators=rf_n_estimators,
                            max_depth=rf_max_depth,
                            min_samples_split=rf_min_samples_split,
                            random_state=random_state,
                            n_jobs=-1,
                        )
                    elif model_name == "LightGBM" and HAS_LGBM:
                        model = LGBMRegressor(objective='regression', random_state=random_state)
                    elif model_name == "XGBoost" and HAS_XGB:
                        model = xgb.XGBRegressor(
                            objective='reg:squarederror',
                            n_estimators=300,
                            random_state=random_state,
                            n_jobs=-1,
                        )
                    else:
                        continue

                    for tr_idx, va_idx in kf.split(X_train_q, y_train_target_q):
                        X_tr, X_va = X_train_q.iloc[tr_idx], X_train_q.iloc[va_idx]
                        y_tr, y_va = y_train_target_q.iloc[tr_idx], y_train_target_q.iloc[va_idx]

                        model.fit(X_tr, y_tr)
                        preds_log = model.predict(X_va)

                        # Metrics on log scale
                        rmse_log = rmse_score(y_va, preds_log)
                        r2 = r2_score(y_va, preds_log)

                        # Convert to dollars (if log used)
                        if do_log_transform:
                            y_va_dollar = np.expm1(y_va)
                            preds_dollar = np.expm1(preds_log)
                        else:
                            y_va_dollar = y_va
                            preds_dollar = preds_log

                        rmse_dollar = rmse_score(y_va_dollar, preds_dollar)

                        fold_rmse_log.append(rmse_log)
                        fold_rmse_dollar.append(rmse_dollar)
                        fold_r2.append(r2)

                    results.append({
                        "Model": model_name,
                        "RMSE (log)": np.mean(fold_rmse_log) if do_log_transform else np.nan,
                        "RMSE ($)": np.mean(fold_rmse_dollar),
                        "R² (log)": np.mean(fold_r2),
                    })

                    # Save importances when available
                    try:
                        if hasattr(model, "feature_importances_"):
                            model.fit(X_train_q, y_train_target_q)
                            feature_importances[model_name] = model.feature_importances_
                    except Exception:
                        pass

                res_df = pd.DataFrame(results).sort_values(by="RMSE ($)")
                st.success("Karşılaştırma tamamlandı.")
                st.dataframe(res_df, use_container_width=True)

                # Plot RMSE ($)
                fig = px.bar(res_df, x="Model", y="RMSE ($)", title="Model Karşılaştırma — RMSE ($)")
                st.plotly_chart(fig, use_container_width=True)

                # Feature importance (tree models)
                if feature_importances:
                    with st.expander("Özellik önemleri (ağaç tabanlı modeller)"):
                        for m, importances in feature_importances.items():
                            imp_df = pd.DataFrame({"feature": X_train_q.columns, "importance": importances}) \
                                .sort_values("importance", ascending=False).head(25)
                            st.write(f"**{m} — en önemli 25 özellik**")
                            st.dataframe(imp_df)
                            st.plotly_chart(px.bar(imp_df, x="feature", y="importance", title=f"{m} Feature Importance"), use_container_width=True)

                # Cache for prediction tab (quick)
                st.session_state["quick_results_df"] = res_df
                st.session_state["X_train_q"] = X_train_q
                st.session_state["X_test_q"] = X_test_q
                st.session_state["y_train_target_q"] = y_train_target_q
                st.session_state["do_log_transform_q"] = do_log_transform
                st.session_state["random_state_q"] = random_state

                # Learning Curve (CV) — pick first chosen model
                st.markdown("### Öğrenme Eğrisi (CV) — Hızlı")
                if chosen:
                    model_for_curve = st.selectbox("Eğri için model seçin", chosen, index=0, key="curve_quick_select")

                    def _build_model_quick(name: str):
                        if name == "Linear Regression":
                            return LinearRegression()
                        if name == "RandomForest":
                            return RandomForestRegressor(
                                n_estimators=rf_n_estimators,
                                max_depth=rf_max_depth,
                                min_samples_split=rf_min_samples_split,
                                random_state=random_state,
                                n_jobs=-1,
                            )
                        if name == "LightGBM" and HAS_LGBM:
                            return LGBMRegressor(objective='regression', random_state=random_state)
                        if name == "XGBoost" and HAS_XGB:
                            return xgb.XGBRegressor(objective='reg:squarederror', n_estimators=300, random_state=random_state, n_jobs=-1)
                        return LinearRegression()

                    mdl_curve = _build_model_quick(model_for_curve)
                    try:
                        sizes, tr_scores, va_scores = learning_curve(
                            mdl_curve, X_train_q, y_train_target_q,
                            train_sizes=[0.1, 0.3, 0.5, 0.7, 1.0],
                            cv=KFold(n_splits=n_splits, shuffle=True, random_state=random_state),
                            scoring='neg_root_mean_squared_error', n_jobs=-1
                        )
                        fig_lc = plt.figure(figsize=(7,4))
                        plt.plot(sizes, -tr_scores.mean(axis=1), marker='o', label='Eğitim RMSE')
                        plt.plot(sizes, -va_scores.mean(axis=1), marker='s', label='Doğrulama RMSE')
                        plt.xlabel('Eğitim Boyutu'); plt.ylabel('RMSE'); plt.legend(); plt.title(f'Öğrenme Eğrisi — {model_for_curve}')
                        st.pyplot(fig_lc, use_container_width=True)
                        plt.close(fig_lc)
                    except Exception as e:
                        st.warning(f"Öğrenme eğrisi oluşturulamadı: {e}")

        st.markdown("---")

        # ===== Advanced Pipeline (Project_final_2) =====
        st.markdown("#### 2) Gelişmiş Boru Hattı (Project_final_2)")
        if PROJ is None:
            st.info("`Project_final_2.py` bulunamadı veya içe aktarılamadı. Bu bölüm dosya olmadan devre dışı.")
        else:
            colA, colB = st.columns([1,1])
            with colA:
                do_stack = st.checkbox("Stacking (LGBM + XGB + CAT)", value=False)
            with colB:
                show_adv_plots = st.checkbox("Artık grafikleri / önem grafikleri", value=True)

            if st.button("🚀 Eğit (Gelişmiş)", type="secondary"):
                with st.spinner("Gelişmiş pipeline eğitiliyor..."):
                    results_adv = PROJ.prepare_and_train(train_df_raw)

                # metrics from results_adv (y_test & predictions inside model?)
                # We'll recompute to display consistently
                mdl = results_adv["model"]
                X_test_s = results_adv.get("X_test_scaled")
                y_test = results_adv.get("y_test")
                if X_test_s is not None and y_test is not None and len(y_test) == len(X_test_s):
                    preds = mdl.predict(X_test_s)
                    rmse = rmse_score(y_test, preds)
                    r2 = r2_score(y_test, preds)
                else:
                    rmse = np.nan; r2 = np.nan

                st.success("Gelişmiş eğitim tamamlandı.")
                st.write({"RMSE (val/log)": (None if np.isnan(rmse) else round(float(rmse),4)), "R² (val/log)": (None if np.isnan(r2) else round(float(r2),4))})

                if do_stack:
                    with st.spinner("Stacking çalışıyor..."):
                        stack_out = PROJ.run_stacking(results_adv, n_folds=5, seed=int(random_state), verbose=False)
                    st.subheader("Stacking Sonuçları (Meta: RidgeCV)")
                    st.write({k: (round(v,4) if isinstance(v,(int,float)) else v) for k,v in stack_out["metrics"].items()})

                # Plots (residuals / feature importance)
                if show_adv_plots:
                    try:
                        y_pred = mdl.predict(results_adv["X_test_scaled"])
                        y_true = results_adv["y_test"]
                        fig1, ax1 = plt.subplots()
                        ax1.hist((y_true - y_pred), bins=40); ax1.set_title("Residuals (log)")
                        st.pyplot(fig1)

                        fig2, ax2 = plt.subplots()
                        ax2.scatter(y_true, y_pred, s=8, alpha=0.6)
                        mn, mx = float(min(np.min(y_true), np.min(y_pred))), float(max(np.max(y_true), np.max(y_pred)))
                        ax2.plot([mn, mx], [mn, mx], "--", linewidth=1)
                        ax2.set_xlabel("Actual (log)"); ax2.set_ylabel("Predicted (log)")
                        st.pyplot(fig2)

                        if hasattr(mdl, "feature_importances_") and results_adv.get("X_train") is not None:
                            imps = mdl.feature_importances_
                            order = np.argsort(imps)[::-1][:25]
                            names = results_adv["X_train"].columns[order]
                            vals = imps[order]
                            fig3, ax3 = plt.subplots(figsize=(6, max(4, 0.3*len(names))))
                            ax3.barh(names[::-1], vals[::-1]); ax3.set_title("Feature Importance (LGBM)")
                            st.pyplot(fig3)
                    except Exception as e:
                        st.warning(f"Görselleştirme hatası: {e}")

                # Cache for prediction tab (advanced)
                st.session_state["adv_results"] = results_adv
                st.session_state["adv_random_state"] = random_state

# -----------------------------
# Prediction Playground
# -----------------------------
with predict_tab:
    st.subheader("Tekil Tahminler ve Model İndir")

    # Which source?
    src = st.radio("Model kaynağı", ["Hızlı (baseline)", "Gelişmiş (Project_final_2)"], index=0, horizontal=True)

    if src == "Hızlı (baseline)":
        if "quick_results_df" not in st.session_state:
            st.info("Önce modelleme sekmesinde **Hızlı** bölümü çalıştırın.")
        else:
            res_df: pd.DataFrame = st.session_state["quick_results_df"]
            X_train_q: pd.DataFrame = st.session_state["X_train_q"]
            X_test_q: pd.DataFrame | None = st.session_state["X_test_q"]
            y_target_q = st.session_state["y_train_target_q"]
            do_log_q = st.session_state["do_log_transform_q"]
            seed_q = st.session_state["random_state_q"]

            best_model_name = res_df.iloc[0]["Model"]
            st.write(f"🔝 En iyi model: **{best_model_name}**")

            # Refit final model on all data
            if best_model_name == "Linear Regression":
                final_model = LinearRegression()
            elif best_model_name == "RandomForest":
                final_model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=2, random_state=seed_q, n_jobs=-1)
            elif best_model_name == "LightGBM" and HAS_LGBM:
                final_model = LGBMRegressor(objective='regression', random_state=seed_q)
            elif best_model_name == "XGBoost" and HAS_XGB:
                final_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=300, random_state=seed_q, n_jobs=-1)
            else:
                final_model = RandomForestRegressor(n_estimators=200, random_state=seed_q, n_jobs=-1)

            final_model.fit(X_train_q, y_target_q)

            # Choose a row
            source = st.radio("Tahmin verisi", options=["Test verisi", "Train verisi"], index=0 if X_test_q is not None else 1)
            if source == "Test verisi" and X_test_q is not None:
                idx = st.slider("Satır index", 0, len(X_test_q)-1, 0)
                row = X_test_q.iloc[[idx]]
            else:
                idx = st.slider("Satır index", 0, len(X_train_q)-1, 0)
                row = X_train_q.iloc[[idx]]

            pred_log = float(final_model.predict(row)[0])
            pred_dollar = float(np.expm1(pred_log)) if do_log_q else float(pred_log)

            st.metric(label="Tahmin Edilen Gelir ($)", value=f"${pred_dollar:,.2f}")
            st.caption("Not: Log dönüşümü açıksa değerler ters log ile $ cinsine çevrilmiştir.")

            # Download trained model
            import joblib
            buf = io.BytesIO()
            joblib.dump({
                "model": final_model,
                "do_log_transform": do_log_q,
                "columns": X_train_q.columns.tolist(),
            }, buf)
            st.download_button("💾 Modeli indir (.joblib)", data=buf.getvalue(), file_name="revenue_model_quick.joblib")

    else:  # Advanced
        if "adv_results" not in st.session_state:
            st.info("Önce modelleme sekmesinde **Gelişmiş** bölümü çalıştırın.")
        else:
            results_adv = st.session_state["adv_results"]
            do_log_adv = True  # Project_final_2 works in log revenue internally

            mdl = results_adv["model"]
            X_train_s = results_adv.get("X_train_scaled")
            X_test_s = results_adv.get("X_test_scaled")

            source = st.radio("Tahmin verisi", options=["Validation (X_test_scaled)", "Training (X_train_scaled)"], index=0 if X_test_s is not None else 1)
            if source.startswith("Validation") and X_test_s is not None:
                idx = st.slider("Satır index", 0, len(X_test_s)-1, 0)
                row = X_test_s[idx:idx+1]
            else:
                idx = st.slider("Satır index", 0, len(X_train_s)-1, 0)
                row = X_train_s[idx:idx+1]

            pred_log = float(mdl.predict(row)[0])
            pred_dollar = float(np.expm1(pred_log)) if do_log_adv else float(pred_log)
            st.metric(label="Tahmin Edilen Gelir ($)", value=f"${pred_dollar:,.2f}")

            # Download (model + scaler + column metadata) — to reload consistently
            import joblib
            buf = io.BytesIO()
            joblib.dump({
                "model": mdl,
                "scaler": results_adv.get("scaler"),
                "columns_train": list(results_adv.get("X_train", pd.DataFrame()).columns),
                "note": "Model log(revenue) üzerinde eğitilmiştir. Çıkış için expm1 uygulayın."
            }, buf)
            st.download_button("💾 Modeli indir (.joblib)", data=buf.getvalue(), file_name="revenue_model_advanced.joblib")

# -----------------------------
# Sidebar Footer / Tips
# -----------------------------
with st.sidebar.expander("ℹ️ Hakkında / İpuçları"):
    st.markdown(
        """
        **Revenue Studio (adapted)**: Train‑only akış varsayılan; test yükleme opsiyoneldir.

        **İpuçları**
        - Veri sadece `train.csv` ile de çalışır. Gelişmiş akış kendi içinde train/validation ayırır.
        - SBERT öneri ilk çalıştırmada model indirdiği için yavaş olabilir; sonraki çalışmalarda cache kullanılır.
        - SHAP için `pip install shap` gerekli ve ağaç tabanlı modellerde etkin.
        - Gelişmiş akışta metrikler log ölçekte raporlanır; dolar cinsine çevirmek için `expm1` kullanın.
        """
    )

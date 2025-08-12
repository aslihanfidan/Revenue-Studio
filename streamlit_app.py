# Streamlit Revenue Studio — Movie Revenue Prediction (Shiny, polished, production-ready)
# ------------------------------------------------------------
# How to run:
#   1) Create a folder (e.g., Streamlit_Proje) and put this file inside as `streamlit_app.py`.
#   2) (Recommended) Create & activate a virtual env.
#   3) Install requirements (see bottom of this file or README):
#        pip install -U streamlit pandas numpy scikit-learn lightgbm xgboost plotly
#   4) Start the app:
#        streamlit run streamlit_app.py
#
# Notes:
# - Works with both old and new scikit-learn versions (RMSE helper below).
# - You can use the built-in example schema or upload your own train/test CSVs.
# - Includes: EDA, robust preprocessing (aligned with your movie dataset),
#   model comparison (LR / RF / LightGBM / XGBoost), cross-validated metrics
#   (log-RMSE, $-RMSE, R^2), feature importance, prediction playground,
#   and model export.

from __future__ import annotations
import os
import io
import ast
import json
import math
import time
import typing as T

import numpy as np
import pandas as pd

import streamlit as st

from inspect import signature
from datetime import datetime

import re
from sentence_transformers import SentenceTransformer

# Sklearn & friends
from sklearn.model_selection import KFold, train_test_split, learning_curve
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
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

# Plotting (native, minimal deps)
import plotly.express as px
import plotly.graph_objects as go
# Matplotlib (for learning curve & SHAP plots)
import matplotlib.pyplot as plt

# SHAP (optional)
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False


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
mode = st.sidebar.radio("Veri kaynağı", ["Örnek Şema + Yükleme", "Sadece Yükleme"], index=0)

st.sidebar.markdown("**Beklenen hedef sütunu:** `revenue` (sayısal)")
with st.sidebar.expander("Yükleme Seçenekleri"):
    train_file = st.file_uploader("train.csv (hedef: revenue)", type=["csv"], key="train")
    test_file = st.file_uploader("test.csv (hedefsiz)", type=["csv"], key="test")

st.sidebar.caption("İstersen sadece bir CSV yükleyip 'Train/Test böl' seçeneğini de kullanabilirsin.")
split_data = st.sidebar.checkbox("Tek CSV yüklersem eğitim/teste böl", value=False)
split_ratio = st.sidebar.slider("Test oranı", 0.1, 0.5, 0.2, 0.05, disabled=not split_data)
random_state = st.sidebar.number_input("Random state", min_value=0, value=42, step=1)

# -----------------------------
# Data Load Helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def _read_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

@st.cache_data(show_spinner=True)
def load_datasets(train_f, test_f, do_split, split_ratio, seed) -> T.Tuple[pd.DataFrame, pd.DataFrame | None]:
    if train_f is not None and not do_split:
        train_df = _read_csv(train_f)
        test_df = _read_csv(test_f) if test_f is not None else None
        return train_df, test_df

    if train_f is not None and do_split:
        full = _read_csv(train_f)
        assert "revenue" in full.columns, "Veride 'revenue' hedef sütunu bulunmalı."
        train_df, valid_df = train_test_split(full, test_size=split_ratio, random_state=seed)
        valid_df = valid_df.drop(columns=["revenue"])  # 'test' gibi kullanılacak (hedefsiz)
        return train_df.reset_index(drop=True), valid_df.reset_index(drop=True)

    # No uploads yet → return empty frames
    return pd.DataFrame(), None

train_df_raw, test_df_raw = load_datasets(train_file, test_file, split_data, split_ratio, random_state)

# -----------------------------
# Main Title
# -----------------------------
st.title("💸 Streamlit Revenue Studio")
st.write("Film gelir tahmini için uçtan uca bir ML uygulaması. Yükle, eğit, karşılaştır, incele ve indir.")

# -----------------------------
# EDA Panel
# -----------------------------
eda_tab, reco_tab, model_tab, predict_tab = st.tabs(
    ["🔎 EDA / Ön İzleme", "🎬 Öneri", "🤖 Modelleme", "🎯 Tahmin Playground"]
)

with eda_tab:
    st.subheader("Veri Önizleme")
    if train_df_raw.empty:
        st.info("Sol taraftan train/test dosyalarını yükleyin veya tek bir CSV yükleyip eğitim/teste bölün.")
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

        # Simple numeric dist plot
        num_cols = train_df_raw.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            col_x = st.selectbox("Sayısal dağılım grafiği için sütun seç", num_cols, index=min(0, len(num_cols)-1))
            fig = px.histogram(train_df_raw, x=col_x, nbins=50, title=f"Dağılım — {col_x}")
            st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Preprocessing aligned with user's movie dataset script
# -----------------------------
LIST_LIKE_COLS_HINT = [
    "genres", "belongs_to_collection", "Keywords", "cast", "crew",
    "production_companies", "production_countries", "spoken_languages"
]
DROP_COLS_DEFAULT = [
    'id', 'belongs_to_collection', 'genres', 'homepage', 'imdb_id', 'original_title',
    'overview', 'poster_path', 'production_companies', 'production_countries',
    'release_date', 'spoken_languages', 'status', 'tagline', 'title', 'Keywords', 'cast', 'crew'
]

@st.cache_data(show_spinner=True)
def preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame | None) -> T.Tuple[pd.DataFrame, pd.DataFrame | None, pd.Series]:
    df_train = train_df.copy()
    df_test = test_df.copy() if test_df is not None else None

    # Basic checks
    assert 'revenue' in df_train.columns, "Train verisinde 'revenue' olmalı."

    # Normalize zeros → NaN for budget/runtime if exist
    for c in ["budget", "runtime"]:
        if c in df_train.columns:
            df_train[c] = df_train[c].replace(0, np.nan)
        if df_test is not None and c in (df_test.columns if df_test is not None else []):
            df_test[c] = df_test[c].replace(0, np.nan)

    # genres fillna and num_genres as in your script
    for c in ["genres"]:
        if c in df_train.columns:
            df_train[c] = df_train[c].fillna('[]')
        if df_test is not None and c in (df_test.columns if df_test is not None else []):
            df_test[c] = df_test[c].fillna('[]')

    def count_from_str(x):
        if pd.isna(x):
            return 0
        try:
            return len(ast.literal_eval(str(x)))
        except Exception:
            return 0

    if "genres" in df_train.columns:
        df_train['num_genres'] = df_train['genres'].apply(count_from_str)
        if df_test is not None:
            df_test['num_genres'] = df_test['genres'].apply(count_from_str)

    # release_date → year, month, day, dow
    if "release_date" in df_train.columns:
        df_train['release_date'] = pd.to_datetime(df_train['release_date'], errors='coerce')
        df_train['release_year'] = df_train['release_date'].dt.year
        df_train['release_month'] = df_train['release_date'].dt.month
        df_train['release_day'] = df_train['release_date'].dt.day
        df_train['release_dayofweek'] = df_train['release_date'].dt.dayofweek
    if df_test is not None and "release_date" in df_test.columns:
        df_test['release_date'] = pd.to_datetime(df_test['release_date'], errors='coerce')
        df_test['release_year'] = df_test['release_date'].dt.year
        df_test['release_month'] = df_test['release_date'].dt.month
        df_test['release_day'] = df_test['release_date'].dt.day
        df_test['release_dayofweek'] = df_test['release_date'].dt.dayofweek

    # fillna median for budget/runtime
    for c in ["budget", "runtime"]:
        if c in df_train.columns:
            med = df_train[c].median()
            df_train[c] = df_train[c].fillna(med)
            if df_test is not None and c in df_test.columns:
                df_test[c] = df_test[c].fillna(med)

    # has_homepage flag
    if "homepage" in df_train.columns:
        df_train['has_homepage'] = df_train['homepage'].apply(lambda x: 1 if pd.notnull(x) and str(x).strip() != '' else 0)
    if df_test is not None and "homepage" in df_test.columns:
        df_test['has_homepage'] = df_test['homepage'].apply(lambda x: 1 if pd.notnull(x) and str(x).strip() != '' else 0)

    # One-hot for original_language — align columns across train/test
    if "original_language" in df_train.columns:
        langs_train = pd.get_dummies(df_train['original_language'], prefix='lang')
        if df_test is not None and "original_language" in df_test.columns:
            langs_test = pd.get_dummies(df_test['original_language'], prefix='lang')
            combined = pd.concat([langs_train, langs_test], axis=0, ignore_index=True).fillna(0)
            langs_train_aligned = combined.iloc[:len(df_train)].reset_index(drop=True)
            langs_test_aligned = combined.iloc[len(df_train):].reset_index(drop=True)
            df_train = pd.concat([df_train.reset_index(drop=True), langs_train_aligned], axis=1)
            df_test = pd.concat([df_test.reset_index(drop=True), langs_test_aligned], axis=1)
        else:
            df_train = pd.concat([df_train, langs_train], axis=1)
        for col in ["original_language"]:
            if col in df_train.columns:
                df_train.drop(columns=[col], inplace=True)
            if df_test is not None and col in df_test.columns:
                df_test.drop(columns=[col], inplace=True)

    # Drop many high-cardinality text columns (same as your script when present)
    for col in DROP_COLS_DEFAULT:
        if col in df_train.columns:
            df_train.drop(columns=[col], inplace=True)
        if df_test is not None and col in (df_test.columns if df_test is not None else []):
            df_test.drop(columns=[col], inplace=True)

    # Final categorical handling on any remaining object cols
    full = pd.concat([df_train.drop(columns=['revenue']), df_test], axis=0, ignore_index=True) if df_test is not None else df_train.drop(columns=['revenue'])
    obj_cols = full.select_dtypes(include=['object']).columns.tolist()
    if obj_cols:
        dummies = pd.get_dummies(full[obj_cols], dummy_na=False)
        full = pd.concat([full.drop(columns=obj_cols), dummies], axis=1)
    else:
        full = full.copy()

    if df_test is not None:
        X_train = full.iloc[:len(df_train)].reset_index(drop=True)
        X_test = full.iloc[len(df_train):].reset_index(drop=True)
    else:
        X_train = full.reset_index(drop=True)
        X_test = None

    y = df_train['revenue'].copy()

    return X_train, X_test, y

# -----------------------------
# Recommendation Tab (SBERT entegre)
# -----------------------------
with reco_tab:
    st.subheader("🎬 Öneri Sistemi (SBERT + Hibrit Skor)")

    if train_df_raw.empty:
        st.info("Önce sol menüden train.csv yükleyin. (Tercihen: title / original_title / overview / genres / cast / crew / release_date sütunları)")
        st.stop()

    # --- Sütun bulucu yardımcı ---
    def _pick_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        low = {c.lower(): c for c in df.columns}
        for c in candidates:
            if c.lower() in low:
                return low[c.lower()]
        return None

    # Veri setinde hangi kolonlar var, tespit et
    title_col    = _pick_col(train_df_raw, ["title","original_title","movie_title","Title"])
    orig_title   = _pick_col(train_df_raw, ["original_title","orig_title"])
    overview_col = _pick_col(train_df_raw, ["overview","description","plot","tagline"])
    tagline_col  = _pick_col(train_df_raw, ["tagline"])
    genres_col   = _pick_col(train_df_raw, ["genres","genre","main_genre"])
    keywords_col = _pick_col(train_df_raw, ["Keywords","keywords"])
    cast_col     = _pick_col(train_df_raw, ["cast","actors"])
    crew_col     = _pick_col(train_df_raw, ["crew","credits"])
    rdate_col    = _pick_col(train_df_raw, ["release_date","date"])

    if not title_col and not orig_title:
        st.error("Film başlığı için 'title' veya 'original_title' bulunmalı.")
        st.stop()

    # Recommender için beklenen kolon isimlerine map’leyip kopya DataFrame oluşturalım
    df_rec_src = train_df_raw.copy()
    def _safe_col(src, col, default):
        return src[col] if col and col in src.columns else default

    df_rec = pd.DataFrame({
        "title":           _safe_col(df_rec_src, title_col, df_rec_src.get(orig_title, pd.Series([""]*len(df_rec_src)))),
        "original_title":  _safe_col(df_rec_src, orig_title, df_rec_src.get(title_col, pd.Series([""]*len(df_rec_src)))),
        "overview":        _safe_col(df_rec_src, overview_col, ""),
        "tagline":         _safe_col(df_rec_src, tagline_col, ""),
        "genres":          _safe_col(df_rec_src, genres_col, "[]"),
        "Keywords":        _safe_col(df_rec_src, keywords_col, "[]"),
        "cast":            _safe_col(df_rec_src, cast_col, "[]"),
        "crew":            _safe_col(df_rec_src, crew_col, "[]"),
        "release_date":    _safe_col(df_rec_src, rdate_col, None),
    })

    # --- Recommender fonksiyonları (özet; canvas’taki tam sürümle uyumlu) ---
    def _safe_list(x):
        if isinstance(x, list): return x
        if isinstance(x, str):
            try: return ast.literal_eval(x)
            except Exception: return []
        return []

    def _set_names(x):
        out=set()
        for q in _safe_list(x):
            if isinstance(q, dict) and q.get("name"):
                out.add(str(q["name"]).strip().lower())
            elif isinstance(q, str):
                out.add(q.strip().lower())
        return out

    def _cast_names(x, top_k=10):
        items=_safe_list(x)
        if items and isinstance(items[0], dict) and 'order' in items[0]:
            items=sorted(items, key=lambda d: d.get('order', 10**9))
        out=[]
        for q in items:
            n = q.get("name") if isinstance(q, dict) else q
            if n: out.append(str(n).strip().lower())
        return set(out[:top_k])

    def _dir_names(x):
        out=set()
        for q in _safe_list(x):
            if isinstance(q, dict):
                if "director" in str(q.get("job","")).lower() and q.get("name"):
                    out.add(str(q["name"]).strip().lower())
            elif isinstance(q, str):
                # metin geldiyse yönetmen adını listeye al
                out.add(q.strip().lower())
        return out

    def _join_words(s): 
        return " ".join(sorted(list(s))).replace(" ","_")

    @st.cache_resource(show_spinner=True)
    def _build_recommender_cached(df_all: pd.DataFrame, model_name="sentence-transformers/all-mpnet-base-v2"):
        m = SentenceTransformer(model_name)
        d = df_all.copy()
        d.fillna({'overview':'','tagline':'','Keywords':'[]','genres':'[]','cast':'[]','crew':'[]'}, inplace=True)
        d["combined_text_rec"] = (
            d["overview"].astype(str) + " " + d.get("tagline","").astype(str) + " " +
            d["genres"].apply(_set_names).apply(_join_words) + " " +
            d["Keywords"].apply(_set_names).apply(_join_words) + " " +
            d["cast"].apply(_cast_names).apply(_join_words) + " " +
            d["crew"].apply(_dir_names).apply(_join_words)
        ).str.strip()
        E = m.encode(d["combined_text_rec"].tolist(), batch_size=64, convert_to_numpy=True, normalize_embeddings=True)
        titles_full = d['title'].fillna(d['original_title']).astype(str).tolist()
        years = pd.to_datetime(d["release_date"], errors="coerce").dt.year.fillna(-1).astype(int).to_numpy()
        title_map={}
        def _norm_title(t):
            t = re.sub(r'\s*\(\d{4}\)\s*$','',str(t))
            t = re.sub(r'\s+',' ',t).strip().lower()
            return t
        for i,t in enumerate(titles_full):
            n=_norm_title(t); title_map.setdefault(n, []).append(i)
        return {
            "df": d, "model": m, "emb": E, "titles_full": titles_full,
            "title_map": title_map, "years": years,
            "genres": d["genres"].apply(_set_names).tolist(),
            "keywords": d["Keywords"].apply(_set_names).tolist(),
            "cast": d["cast"].apply(_cast_names).tolist(),
            "dirs": d["crew"].apply(_dir_names).tolist()
        }

    def _apply_rank_postfix(score, titles_full, idx_query,
                            franchise_tokens=('batman','joker','dc'),
                            franchise_penalty=0.05,
                            exclude_same_title=True):
        def _norm_title_simple(t):
            t = re.sub(r'\s*\(\d{4}\)\s*$','',str(t)); t = re.sub(r'\s+',' ',t).strip().lower(); return t
        titles_norm = [_norm_title_simple(t) for t in titles_full]
        if idx_query is not None and exclude_same_title:
            t0 = titles_norm[idx_query]
            same_mask = np.array([(_norm_title_simple(t) == t0) for t in titles_full], dtype=bool)
            score[same_mask] = -1.0; score[idx_query] = -1.0
        if idx_query is not None:
            q_title = titles_norm[idx_query]
            if any(tok in q_title for tok in franchise_tokens):
                penalize_mask = np.array([any(tok in t for tok in franchise_tokens) for t in titles_norm], dtype=bool)
                score[penalize_mask] -= franchise_penalty
        return score

    def _jac(a,b):
        u=(a|b); return len(a&b)/len(u) if u else 0.0

    def recommend_by_title(title, rec, top_k=5,
                           w_sbert=0.45, w_kw=0.30, w_dir=0.15, w_cast=0.10,
                           force_same_genre=False, min_shared_keywords=0,
                           prefer_genres=("drama","mystery","thriller"),
                           year_boost=0.02, year_window=7,
                           title_match_min_sim=0.45,
                           franchise_tokens=('batman','joker','dc'),
                           franchise_penalty=0.05):
        df=rec["df"]; E=rec["emb"]; titles=rec["titles_full"]; years=rec["years"]
        G=rec["genres"]; K=rec["keywords"]; C=rec["cast"]; D=rec["dirs"]
        model=rec["model"]; tmap=rec.get("title_map",{})
        def _norm_title_simple(t):
            t = re.sub(r'\s*\(\d{4}\)\s*$','',str(t)); t = re.sub(r'\s+',' ',t).strip().lower(); return t
        ntitle=_norm_title_simple(title)
        if ntitle in tmap: idx=tmap[ntitle][0]
        else:
            mask = df['title'].fillna('').str.contains(ntitle, case=False, regex=False) | df['original_title'].fillna('').str.contains(ntitle, case=False, regex=False)
            idxs = df.index[mask].tolist()
            idx = (idxs[0] if idxs else None)
            if idx is None:
                if rec.get("title_emb") is None:
                    rec["title_emb"] = model.encode(titles, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)
                qv = model.encode([str(title)], convert_to_numpy=True, normalize_embeddings=True)[0]
                sims_t = rec["title_emb"] @ qv
                j = int(np.argmax(sims_t))
                if float(sims_t[j]) < title_match_min_sim:
                    raise ValueError(f"Title «{title}» not found (nearest: {titles[j]}, sim={float(sims_t[j]):.2f}).")
                idx = j
        sims_sbert = E @ E[idx]
        sims_kw    = np.array([_jac(K[idx], K[j]) for j in range(len(df))])
        sims_cast  = np.array([_jac(C[idx], C[j]) for j in range(len(df))])
        sims_dir   = np.array([_jac(D[idx], D[j]) for j in range(len(df))])
        score = 0.45*sims_sbert + 0.30*sims_kw + 0.15*sims_dir + 0.10*sims_cast
        pg=set([g.strip().lower() for g in (prefer_genres or [])])
        if pg: score += 0.05*np.array([len(pg & G[j]) for j in range(len(df))])
        mask_valid = np.ones(len(df), dtype=bool)
        if force_same_genre: mask_valid &= np.array([len(G[idx] & G[j]) > 0 for j in range(len(df))])
        if min_shared_keywords > 0: mask_valid &= np.array([len(K[idx] & K[j]) >= min_shared_keywords for j in range(len(df))])
        y0 = years[idx]
        if y0 != -1 and year_window > 0:
            delta = np.abs(years - y0)
            score += 0.02 * np.clip((year_window - delta) / year_window, 0, 1)
        score = _apply_rank_postfix(score, titles_full=titles, idx_query=idx,
                                    franchise_tokens=franchise_tokens,
                                    franchise_penalty=franchise_penalty,
                                    exclude_same_title=True)
        score[~mask_valid] = -1.0
        order = np.argsort(-score)[:top_k]
        return pd.DataFrame({"title":[titles[j] for j in order], "similarity":[float(score[j]) for j in order]})

    def recommend_by_title_nonfranchise(title, rec, top_k=5,
                                        franchise_tokens=('batman','joker','gotham','arkham','dark knight'),
                                        **kwargs):
        all_recs = recommend_by_title(title, rec, top_k=50,
                                      franchise_tokens=franchise_tokens,
                                      franchise_penalty=0.0, **kwargs)
        patt = '|'.join([re.escape(t) for t in franchise_tokens]) if franchise_tokens else None
        if patt:
            mask = ~all_recs['title'].str.lower().str.contains(patt, na=False, regex=True)
            return all_recs[mask].head(top_k).reset_index(drop=True)
        return all_recs.head(top_k).reset_index(drop=True)

    def recommend_hybrid(title, rec, top_k=5, nonfr_k=5,
                         franchise_tokens=('batman','joker','gotham','arkham','dark knight'), **kwargs):
        nonfr = recommend_by_title_nonfranchise(title, rec, top_k=nonfr_k,
                                                franchise_tokens=franchise_tokens, **kwargs)
        has_fr = False
        if len(nonfr):
            patt = '|'.join([re.escape(t) for t in franchise_tokens]) if franchise_tokens else None
            if patt:
                has_fr = nonfr['title'].str.lower().str.contains(patt, na=False, regex=True).any()
        if (not has_fr) and franchise_tokens:
            full = recommend_by_title(title, rec, top_k=10,
                                      franchise_tokens=franchise_tokens,
                                      franchise_penalty=0.01, **kwargs)
            patt = '|'.join([re.escape(t) for t in franchise_tokens])
            for _, row in full.iterrows():
                t = str(row['title']).lower()
                if re.search(patt, t):
                    nonfr.loc[len(nonfr)] = row
                    break
        return nonfr.sort_values('similarity', ascending=False).head(top_k).reset_index(drop=True)

    def recommend_by_query(query, rec, top_k=5, prefer_genres=("science fiction","thriller","mystery"),
                           k_wide=2000, genre_boost=0.05):
        model, E = rec["model"], rec["emb"]; titles = rec["titles_full"]; G = rec["genres"]
        qv = model.encode([str(query)], convert_to_numpy=True, normalize_embeddings=True)[0]
        sims = E @ qv
        cand = np.argsort(-sims)[:k_wide]
        pg = set([g.strip().lower() for g in (prefer_genres or [])])
        scores=[]
        for j in cand:
            bump = genre_boost * len(pg & G[j]) if pg else 0.0
            scores.append((j, float(sims[j]) + bump))
        scores.sort(key=lambda x: x[1], reverse=True)
        top=[j for j,_ in scores[:top_k]]
        return pd.DataFrame({"title":[titles[j] for j in top], "similarity":[float(sims[j]) for j in top]})

    # --- Modeli kur / cache’ten çek
    with st.spinner("Öneri modeli hazırlanıyor (SBERT gömlemeleri çıkarılıyor)..."):
        rec_obj = _build_recommender_cached(df_rec)

    # --- UI
    st.caption("Bir film seçin ya da serbest metin yazın; franchise filtrelemeyi ayarlayın.")
    colA, colB = st.columns([2, 1])
    with colA:
        titles = df_rec["title"].fillna(df_rec["original_title"]).astype(str).dropna().drop_duplicates().sort_values().tolist()
        film_sec = st.selectbox("Film seç (title-based / hybrid için):", titles, index=0)
    with colB:
        top_k = st.slider("Öneri sayısı", 3, 10, 5)

    default_fr_tokens = "batman,joker,gotham,arkham,dark knight"
    franchise_tokens_in = st.text_input("Franchise filtre anahtarları (virgülle):", value=default_fr_tokens)
    franchise_tokens = tuple([t.strip().lower() for t in franchise_tokens_in.split(",") if t.strip()])

    # Tür tercihleri (varsa)
    # df_rec['genres'] list/str olduğu için düz isim seti çıkaralım
    def _all_genre_names(s):
        names=set()
        for x in s:
            for n in _set_names(x):
                names.add(n)
        return sorted(list(names))
    try:
        all_genres = _all_genre_names(df_rec["genres"])
    except Exception:
        all_genres = []
    prefer_genres = st.multiselect("Tercih edilen tür(ler)i vurgula", options=all_genres, default=["drama","mystery","thriller"] if "drama" in all_genres else [])

    query_text = st.text_input("Serbest metin (ör. 'space survival thriller')", value="dark gritty crime thriller")

    t1, t2, t3, t4 = st.tabs(["Title-based", "Non-franchise", "Query-based", "Hybrid"])

    with t1:
        try:
            out = recommend_by_title(
                film_sec, rec_obj, top_k=top_k,
                prefer_genres=tuple(prefer_genres) if prefer_genres else (),
                franchise_tokens=franchise_tokens, franchise_penalty=0.05
            )
            st.dataframe(out, use_container_width=True)
        except Exception as e:
            st.warning(f"Başlık bazlı öneri başarısız: {e}")

    with t2:
        try:
            out = recommend_by_title_nonfranchise(
                film_sec, rec_obj, top_k=top_k,
                franchise_tokens=franchise_tokens,
                prefer_genres=tuple(prefer_genres) if prefer_genres else ()
            )
            st.dataframe(out, use_container_width=True)
        except Exception as e:
            st.warning(f"Non-franchise öneri başarısız: {e}")

    with t3:
        try:
            if not query_text.strip():
                st.info("Bir sorgu yazın (ör. 'space survival thriller').")
            else:
                out = recommend_by_query(
                    query_text, rec_obj, top_k=top_k,
                    prefer_genres=tuple(prefer_genres) if prefer_genres else ()
                )
                st.dataframe(out, use_container_width=True)
        except Exception as e:
            st.warning(f"Query-based öneri başarısız: {e}")

    with t4:
        try:
            out = recommend_hybrid(
                film_sec, rec_obj, top_k=top_k, nonfr_k=top_k,
                franchise_tokens=franchise_tokens,
                prefer_genres=tuple(prefer_genres) if prefer_genres else ()
            )
            st.dataframe(out, use_container_width=True)
        except Exception as e:
            st.warning(f"Hybrid öneri başarısız: {e}")

# -----------------------------
# Modeling Tab
# -----------------------------
with model_tab:
    st.subheader("Model Eğitimi ve Karşılaştırma")

    if train_df_raw.empty:
        st.info("Önce veri yükleyin.")
    else:
        do_log_transform = st.checkbox("Gelir (revenue) için log1p dönüşümü uygula", value=True)
        n_splits = st.slider("K-Fold (CV) kat sayısı", 3, 10, 5)

        models_available = {
            "Linear Regression": True,
            "RandomForest": True,
            "LightGBM": HAS_LGBM,
            "XGBoost": HAS_XGB,
        }
        chosen = [m for m, ok in models_available.items() if ok and st.checkbox(m, value=(m in ["RandomForest", "LightGBM"]))]
        if not chosen:
            st.warning("En az bir model seçin.")

        # RF hyperparams (quick)
        with st.expander("RandomForest Hiperparametreleri"):
            rf_n_estimators = st.slider("n_estimators", 50, 500, 200, 50)
            rf_max_depth = st.selectbox("max_depth", options=[None, 5, 10, 15, 20], index=2)
            rf_min_samples_split = st.selectbox("min_samples_split", options=[2, 5, 10], index=0)

        # Preprocess
        X_train, X_test, y = preprocess(train_df_raw, test_df_raw)

        # Optional log transform
        if do_log_transform:
            y_train_target = np.log1p(y)
        else:
            y_train_target = y

        st.write(f"**Eğitim boyutu:** {X_train.shape} | **Hedef:** {y_train_target.shape}")

        if st.button("🚀 Eğit & Karşılaştır", type="primary"):
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

                    for tr_idx, va_idx in kf.split(X_train, y_train_target):
                        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
                        y_tr, y_va = y_train_target.iloc[tr_idx], y_train_target.iloc[va_idx]

                        model.fit(X_tr, y_tr)
                        preds_log = model.predict(X_va)

                        # Metrics on log scale
                        rmse_log = rmse_score(y_va, preds_log)
                        r2 = r2_score(y_va, preds_log)

                        # Convert to dollars (if log used) for interpretability
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
                            model.fit(X_train, y_train_target)
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
                            imp_df = pd.DataFrame({"feature": X_train.columns, "importance": importances})\
                                .sort_values("importance", ascending=False).head(25)
                            st.write(f"**{m} — en önemli 25 özellik**")
                            st.dataframe(imp_df)
                            st.plotly_chart(px.bar(imp_df, x="feature", y="importance", title=f"{m} Feature Importance"), use_container_width=True)

                st.session_state["model_results_df"] = res_df
                st.session_state["X_train"] = X_train
                st.session_state["X_test"] = X_test
                st.session_state["y_train_target"] = y_train_target
                st.session_state["do_log_transform"] = do_log_transform
                st.session_state["random_state"] = random_state

                # === Learning Curve (CV) ===
                st.markdown("### Öğrenme Eğrisi (CV)")
                if chosen:
                    model_for_curve = st.selectbox("Eğri için model seçin", chosen, index=0)
                    # Build the selected model (same mapping as above)
                    def _build_model(name: str):
                        if name == "Linear Regression":
                            return LinearRegression()
                        if name == "RandomForest":
                            return RandomForestRegressor(
                                n_estimators={{}} if 'rf_n_estimators' not in locals() else rf_n_estimators,
                                max_depth=None if 'rf_max_depth' not in locals() else rf_max_depth,
                                min_samples_split=2 if 'rf_min_samples_split' not in locals() else rf_min_samples_split,
                                random_state=random_state,
                                n_jobs=-1,
                            )
                        if name == "LightGBM" and HAS_LGBM:
                            return LGBMRegressor(objective='regression', random_state=random_state)
                        if name == "XGBoost" and HAS_XGB:
                            return xgb.XGBRegressor(objective='reg:squarederror', n_estimators=300, random_state=random_state, n_jobs=-1)
                        return LinearRegression()

                    mdl_curve = _build_model(model_for_curve)
                    try:
                        sizes, tr_scores, va_scores = learning_curve(
                            mdl_curve, X_train, y_train_target,
                            train_sizes=[0.1, 0.3, 0.5, 0.7, 1.0],
                            cv=kf, scoring='neg_root_mean_squared_error', n_jobs=-1
                        )
                        fig_lc = plt.figure(figsize=(7,4))
                        plt.plot(sizes, -tr_scores.mean(axis=1), marker='o', label='Eğitim RMSE')
                        plt.plot(sizes, -va_scores.mean(axis=1), marker='s', label='Doğrulama RMSE')
                        plt.xlabel('Eğitim Boyutu'); plt.ylabel('RMSE'); plt.legend(); plt.title(f'Öğrenme Eğrisi — {model_for_curve}')
                        st.pyplot(fig_lc, use_container_width=True)
                        plt.close(fig_lc)
                    except Exception as e:
                        st.warning(f"Öğrenme eğrisi oluşturulamadı: {e}")

                # === SHAP Figures ===
                st.markdown("### SHAP (Özellik Etkileri)")
                if 'best_model_name' in locals():
                    shap_model_name = st.selectbox("SHAP için model seçin", [res['Model'] for res in results], index=0)
                else:
                    shap_model_name = st.selectbox("SHAP için model seçin", chosen, index=0) if chosen else None

                tree_like = {"RandomForest", "LightGBM", "XGBoost"}
                if shap_model_name is None:
                    st.info("Önce model eğitimi tamamlayın.")
                elif not HAS_SHAP:
                    st.info("SHAP kütüphanesi yüklü değil. `pip install shap` ile kurabilirsiniz.")
                elif shap_model_name not in tree_like:
                    st.info("SHAP bu demoda ağaç tabanlı modeller için etkinleştirildi (RF/LightGBM/XGBoost).")
                else:
                    # Rebuild selected model and fit on full training data
                    def _build_tree_model(name: str):
                        if name == "RandomForest":
                            return RandomForestRegressor(
                                n_estimators={{}} if 'rf_n_estimators' not in locals() else rf_n_estimators,
                                max_depth=None if 'rf_max_depth' not in locals() else rf_max_depth,
                                min_samples_split=2 if 'rf_min_samples_split' not in locals() else rf_min_samples_split,
                                random_state=random_state, n_jobs=-1
                            )
                        if name == "LightGBM" and HAS_LGBM:
                            return LGBMRegressor(objective='regression', random_state=random_state)
                        if name == "XGBoost" and HAS_XGB:
                            return xgb.XGBRegressor(objective='reg:squarederror', n_estimators=300, random_state=random_state, n_jobs=-1)
                        return None

                    mdl = _build_tree_model(shap_model_name)
                    if mdl is None:
                        st.warning("Seçilen model yeniden oluşturulamadı.")
                    else:
                        try:
                            mdl.fit(X_train, y_train_target)
                            # Subsample for speed
                            n_samp = int(min(1000, len(X_train)))
                            Xs = X_train.sample(n_samp, random_state=random_state)
                            explainer = shap.TreeExplainer(mdl)
                            shap_values = explainer.shap_values(Xs)
                            # Summary plot
                            fig_shap = plt.figure(figsize=(7,4))
                            shap.summary_plot(shap_values, Xs, show=False)
                            st.pyplot(fig_shap, use_container_width=True)
                            plt.close(fig_shap)
                        except Exception as e:
                            st.warning(f"SHAP grafikleri oluşturulamadı: {e }")


# -----------------------------
# Prediction Playground
# -----------------------------
with predict_tab:
    st.subheader("Tekil Tahminler ve Model İndir")

    if "model_results_df" not in st.session_state:
        st.info("Önce modelleme sekmesinde eğitim/karşılaştırma yapın.")
    else:
        res_df: pd.DataFrame = st.session_state["model_results_df"]
        X_train: pd.DataFrame = st.session_state["X_train"]
        X_test: pd.DataFrame | None = st.session_state["X_test"]
        y_target = st.session_state["y_train_target"]
        do_log_transform = st.session_state["do_log_transform"]
        seed = st.session_state["random_state"]

        best_model_name = res_df.iloc[0]["Model"]
        st.write(f"🔝 En iyi model: **{best_model_name}**")

        # Refit final model on all data
        if best_model_name == "Linear Regression":
            final_model = LinearRegression()
        elif best_model_name == "RandomForest":
            final_model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=2, random_state=seed, n_jobs=-1)
        elif best_model_name == "LightGBM" and HAS_LGBM:
            final_model = LGBMRegressor(objective='regression', random_state=seed)
        elif best_model_name == "XGBoost" and HAS_XGB:
            final_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=300, random_state=seed, n_jobs=-1)
        else:
            final_model = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)

        final_model.fit(X_train, y_target)

        # Choose a row to predict from X_test if available; otherwise random from X_train
        source = st.radio("Tahmin verisi", options=["Test verisi", "Train verisi"], index=0 if X_test is not None else 1)
        if source == "Test verisi" and X_test is not None:
            idx = st.slider("Satır index", 0, len(X_test)-1, 0)
            row = X_test.iloc[[idx]]
        else:
            idx = st.slider("Satır index", 0, len(X_train)-1, 0)
            row = X_train.iloc[[idx]]

        pred_log = float(final_model.predict(row)[0])
        pred_dollar = float(np.expm1(pred_log)) if do_log_transform else float(pred_log)

        st.metric(label="Tahmin Edilen Gelir ($)", value=f"${pred_dollar:,.2f}")
        st.caption("Not: Log dönüşümü açıksa değerler ters log ile $ cinsine çevrilmiştir.")

        # Download trained model
        import joblib
        buf = io.BytesIO()
        joblib.dump({
            "model": final_model,
            "do_log_transform": do_log_transform,
            "columns": X_train.columns.tolist(),
        }, buf)
        st.download_button("💾 Modeli indir (.joblib)", data=buf.getvalue(), file_name="revenue_model.joblib")

# -----------------------------
# Footer / About
# -----------------------------
with st.sidebar.expander("ℹ️ Hakkında / İpuçları"):
    st.markdown(
        """
        **Revenue Studio**: Film gelir tahmini için pratik ve esnek bir uygulama.

        **İpuçları**
        - Eğer tek bir CSV yüklerseniz ve `revenue` içeriyorsa, *Tek CSV'yi böl* seçeneği ile hızlıca deneme yapabilirsiniz.
        - Kaynak veri film veri setiniz değilse de çalışır; `revenue` hedefi ve sayısal/kategorik sütunlarla uyumludur.
        - LightGBM/XGBoost yüklü değilse işaretlemeyin ya da `pip install lightgbm xgboost` ile kurun.
        - RMSE eski sklearn sürümlerinde de güvenlidir (özel helper ile).
        """
    )

# -----------------------------
# (Optional) Requirements quick list
# -----------------------------
# REQUIREMENTS = """
# streamlit>=1.35
# pandas>=1.5
# numpy>=1.23
# scikit-learn>=0.22
# lightgbm>=4.0
# xgboost>=1.7
# plotly>=5.20
# joblib>=1.3
# """
#
# with st.expander("📦 requirements.txt (kopyala-yapıştır)"):
#     st.code(REQUIREMENTS, language="text")

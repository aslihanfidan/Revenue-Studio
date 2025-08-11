

##############################################################################
# # Keşifçi Veri Analizi
##############################################################################
# =========================
# Imports & Settings
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, re, json, ast
from collections import defaultdict
from datetime import datetime
from sklearn.model_selection import train_test_split, KFold, learning_curve
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import RidgeCV
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import shap

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.max_rows', 50)

# =========================
# Utility helpers
# =========================
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.select_dtypes(include='number').quantile([0,0.05,0.50,0.95,0.99,1]).T)

def safe_eval_entities(val, max_unwrap=3):
    if val is None or (isinstance(val, float) and np.isnan(val)): return []
    if isinstance(val, str):
        s = val.strip().lower()
        if s in {"", "none", "null", "nan", "[]", "{}", "unknown"}: return []
    if isinstance(val, list): return [x for x in val if isinstance(x, dict)]
    if isinstance(val, dict): return [val]
    if isinstance(val, str):
        cur = val.strip(); obj=None
        for _ in range(max_unwrap):
            try: obj = json.loads(cur)
            except Exception:
                try: obj = ast.literal_eval(cur)
                except Exception:
                    cur_norm = re.sub(r",\s*([}\]])", r"\1", cur.replace("'", '"'))
                    try: obj = json.loads(cur_norm)
                    except Exception: obj=None
            if isinstance(obj, str): cur=obj.strip(); continue
            break
        if isinstance(obj, dict): return [obj]
        if isinstance(obj, list): return [x for x in obj if isinstance(x, dict)]
        return []
    return []

def names_from_entities(val):
    items = safe_eval_entities(val); out=[]
    for d in items:
        name = d.get("name")
        if name: out.append(str(name).strip())
    return out

def get_top_entities(df_in, column, job_filter=None, top_n=50, min_count=5,
                     use_target=False, target_col="revenue",
                     w_count=0.7, w_pop=0.3, w_target=0.7):
    stats = defaultdict(lambda: {"target_sum":0.0,"popularity_sum":0.0,"count":0})
    for _, row in df_in.iterrows():
        entities = safe_eval_entities(row.get(column, '[]'))
        for ent in entities:
            name = ent.get("name")
            if not name: continue
            if job_filter and ent.get("job") != job_filter: continue
            s = stats[name]
            if use_target: s["target_sum"] += float(row.get(target_col, 0.0))
            s["popularity_sum"] += float(row.get("popularity", 0.0))
            s["count"] += 1
    scored=[]
    for name, st in stats.items():
        if st["count"] < min_count: continue
        avg_pop = st["popularity_sum"]/st["count"] if st["count"] else 0.0
        if use_target:
            avg_tgt = st["target_sum"]/st["count"] if st["count"] else 0.0
            score = w_target*avg_tgt + (1-w_target)*avg_pop
        else:
            score = w_count*st["count"] + w_pop*avg_pop
        scored.append((name, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return set(x[0] for x in scored[:top_n])

def normalize_name(name): return str(name).strip().lower()
def winsorize_by_quantile(s, lower_q=0.05, upper_q=0.95):
    lo = s.quantile(lower_q); hi = s.quantile(upper_q)
    return s.clip(lower=lo, upper=hi)

# =========================
# Feature Engineering
# =========================
def feature_engineering(df, top_genres, top_actors, top_directors, top_companies, top_countries):
    df = df.copy()
    df["release_date"]  = pd.to_datetime(df["release_date"], errors="coerce", format="%m/%d/%y")
    df["release_year"]  = df["release_date"].dt.year
    df["release_month"] = df["release_date"].dt.month

    genres_p  = df["genres"].apply(safe_eval_entities)
    cast_p    = df["cast"].apply(safe_eval_entities)
    crew_p    = df["crew"].apply(safe_eval_entities)
    comp_p    = df["production_companies"].apply(safe_eval_entities)
    country_p = df["production_countries"].apply(safe_eval_entities)

    df["main_genre"]      = genres_p.apply(lambda lst: lst[0]["name"] if lst else "Unknown")
    df["num_genres"]      = genres_p.apply(len)
    df["has_top_genre"]   = genres_p.apply(lambda lst: int(any(normalize_name(g.get("name")) in top_genres for g in lst)))
    df["top_actor_count"] = cast_p.apply(lambda lst: sum(1 for i in lst if normalize_name(i.get("name")) in top_actors))
    df["has_top_director"]= crew_p.apply(lambda lst: int(any(i.get("job")=="Director" and normalize_name(i.get("name")) in top_directors for i in lst)))
    df["top_company_count"]= comp_p.apply(lambda lst: sum(1 for i in lst if normalize_name(i.get("name")) in top_companies))
    df["has_top_country"] = country_p.apply(lambda lst: int(any(normalize_name(i.get("name")) in top_countries for i in lst)))
    df["is_sequel"] = df["title"].apply(lambda x: int(bool(re.search(r"(Part|II|III|IV|2|3|4)", str(x), re.IGNORECASE))))
    df["is_english"] = (df["original_language"] == "en").astype(int)
    top_langs = df["original_language"].value_counts().head(10).index if 'original_language' in df.columns else []
    for lang in top_langs: df[f"lang_{lang}"] = (df["original_language"] == lang).astype(int)
    return df

def create_combined_text(df):
    def process_json_column_to_names(col):
        items = safe_eval_entities(col)
        if isinstance(items, list):
            return " ".join([str(i.get("name","")).strip().replace(" ","_") for i in items if i.get("name")])
        return ""
    def process_cast_to_names(col, top_k=10):
        items = safe_eval_entities(col)
        if isinstance(items, list) and items:
            if isinstance(items[0], dict) and 'order' in items[0]:
                items = sorted(items, key=lambda d: d.get('order', 10**9))
            names = [str(i.get("name","")).strip().replace(" ","_") for i in items if i.get("name")]
            return " ".join(names[:top_k])
        return ""
    def process_crew_to_names(col, keep_jobs=None, top_k=10):
        if keep_jobs is None:
            keep_jobs = {"Director","Writer","Screenplay","Producer","Executive Producer","Editor","Cinematography","Original Music Composer","Music"}
        items = safe_eval_entities(col)
        if isinstance(items, list):
            names=[]
            for i in items:
                if not isinstance(i, dict): continue
                job=i.get("job"); name=i.get("name")
                if name and (job in keep_jobs): names.append(str(name).strip().replace(" ","_"))
            return " ".join(names[:top_k])
        return ""
    df = df.copy()
    df['overview'] = df['overview'].fillna('')
    df['tagline']  = df['tagline'].fillna('') if 'tagline' in df.columns else ''
    df['genres_str']   = df['genres'].apply(process_json_column_to_names) if 'genres' in df.columns else ''
    kw_col = 'Keywords' if 'Keywords' in df.columns else ('keywords' if 'keywords' in df.columns else None)
    df['keywords_str'] = df[kw_col].apply(process_json_column_to_names) if kw_col else ''
    df['cast_str']     = df['cast'].apply(process_cast_to_names) if 'cast' in df.columns else ''
    df['crew_str']     = df['crew'].apply(process_crew_to_names) if 'crew' in df.columns else ''
    df['combined_text'] = (df['overview'].astype(str)+" "+df['genres_str'].astype(str)+" "+df['keywords_str'].astype(str)+" "+df['cast_str'].astype(str)+" "+df['crew_str'].astype(str)+" "+df['tagline'].astype(str)).str.strip()
    return df

# TF-IDF + SVD
def compute_sbert_embeddings(df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    texts = df["combined_text"].fillna("").astype(str).tolist()
    tfidf = TfidfVectorizer(min_df=5, ngram_range=(1,2), max_features=150_000)
    X_tfidf = tfidf.fit_transform(texts)
    svd = TruncatedSVD(n_components=400, random_state=42)
    X_svd = svd.fit_transform(X_tfidf)
    emb_cols = [f"sbert_{i}" for i in range(X_svd.shape[1])]
    df_svd = pd.DataFrame(X_svd, columns=emb_cols, index=df.index)
    return pd.concat([df, df_svd], axis=1).copy()

# =========================
# Pipeline: prepare & train
# =========================
def prepare_and_train(df):
    df = df.copy()
    df.drop(columns=['homepage'], inplace=True, errors='ignore')
    fill_str = {
        'genres':'[]','production_companies':'[]','production_countries':'[]',
        'spoken_languages':'[]','Keywords':'[]','cast':'[]','crew':'[]',
        'overview':'No overview available','poster_path':'No poster','status':'Released'
    }
    df.fillna(value=fill_str, inplace=True)
    df['runtime'] = df['runtime'].fillna(df['runtime'].median())

    top_genres     = {normalize_name(n) for n in get_top_entities(df, "genres", top_n=10, min_count=5, use_target=False)}
    top_actors     = {normalize_name(n) for n in get_top_entities(df, "cast",   top_n=50, min_count=5, use_target=False)}
    top_directors  = {normalize_name(n) for n in get_top_entities(df, "crew", job_filter="Director", top_n=30, min_count=3, use_target=False)}
    top_companies  = {normalize_name(n) for n in get_top_entities(df, "production_companies", top_n=30, min_count=3, use_target=False)}
    top_countries  = {normalize_name(n) for n in get_top_entities(df, "production_countries", top_n=15, min_count=3, use_target=False)}

    genres_parsed = df['genres'].apply(safe_eval_entities)
    df['main_genre'] = genres_parsed.apply(lambda lst: (lst[0].get('name') if lst and lst[0].get('name') else 'Unknown'))

    df['has_budget'] = (df['budget'] > 0).astype(int)
    global_budget_median = df.loc[df['budget'] > 0, 'budget'].median()
    genre_budget_median = (df.loc[df['budget'] > 0].groupby('main_genre')['budget'].median())
    df['budget'] = np.where(df['budget'] > 0, df['budget'], df['main_genre'].map(genre_budget_median).fillna(global_budget_median))
    df['budget'] = winsorize_by_quantile(df['budget'], 0.02, 0.90).astype(float)
    df['popularity'] = winsorize_by_quantile(df['popularity'], 0.00, 0.90).astype(float)
    df['budget_log']  = np.log1p(df['budget'])
    df['revenue_log'] = np.log1p(df['revenue'])
    df['popularity_log'] = np.log1p(df['popularity'])

    df_eng = feature_engineering(df, top_genres, top_actors, top_directors, top_companies, top_countries)
    df_eng = create_combined_text(df_eng)
    df_emb = compute_sbert_embeddings(df_eng)

    drop_cols = [
        'id','imdb_id','original_title','overview','tagline','poster_path',
        'belongs_to_collection','genres','Keywords','cast','crew',
        'combined_text','genres_str','keywords_str','cast_str','crew_str','status',
        'revenue','production_companies','production_countries','spoken_languages',
        'title','release_date','budget','popularity'
    ]
    df_model = df_emb.drop(columns=[c for c in drop_cols if c in df_emb.columns], errors='ignore')
    if "main_genre" in df_model.columns and df_model['main_genre'].dtype == 'O':
        df_model = pd.get_dummies(df_model, columns=["main_genre"], drop_first=True, dtype=int)
    if 'original_language' in df_model.columns:
        df_model = df_model.drop(columns=['original_language'])

    y = df_emb['revenue_log']
    sbert_cols = [c for c in df_model.columns if c.startswith("sbert_")]
    X = df_model.drop(columns=['revenue','revenue_cpi','revenue_log'], errors='ignore')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    scale_cols = ['budget_log','popularity_log','runtime','num_genres','release_year','release_month','top_actor_count','top_company_count'] + sbert_cols
    scale_cols = [c for c in scale_cols if c in X_train.columns]
    scaler = StandardScaler()
    X_train_scaled = X_train.copy(); X_test_scaled = X_test.copy()
    X_train_scaled[scale_cols] = scaler.fit_transform(X_train_scaled[scale_cols])
    X_test_scaled[scale_cols]  = scaler.transform(X_test_scaled[scale_cols])

    model = LGBMRegressor(n_estimators=1200, learning_rate=0.05, num_leaves=63, min_child_samples=30,
                          subsample=0.8, colsample_bytree=0.9, random_state=42, verbose=-1)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    rmse  = mean_squared_error(y_test, y_pred, squared=False)
    mae   = mean_absolute_error(y_test, y_pred)
    r2    = r2_score(y_test, y_pred)
    rmsle = np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_test))**2))
    mape  = mean_absolute_percentage_error(y_test, y_pred) * 100
    smape = 100*np.mean(2*np.abs(y_pred - y_test)/(np.abs(y_test)+np.abs(y_pred)))
    print("LightGBM (TFIDF+SVD + engineered)")
    print(f"RMSE : {rmse:.4f}\nMAE  : {mae:.4f}\nR²   : {r2:.4f}\nRMSLE: {rmsle:.4f}\nMAPE : {mape:.2f}%\nSMAPE: {smape:.2f}%")

    return {'model':model,'scaler':scaler,'X_train':X_train,'X_test':X_test,'y_train':y_train,'y_test':y_test,
            'X_train_scaled':X_train_scaled,'X_test_scaled':X_test_scaled,'df_all':df_emb}

# =========================
# Stacking (OOF) + Meta RidgeCV
# =========================
def _rmse(y_true, y_pred): return float(np.sqrt(mean_squared_error(y_true, y_pred)))
def _align_test_columns(X_train, X_test): return X_test.reindex(columns=X_train.columns, fill_value=0.0)

def oof_model(model_in, Xtr, ytr, Xte, kf, fit_params=None, name="model"):
    n = Xtr.shape[0]; oof_pred = np.zeros(n, dtype=float); test_preds=[]
    for fold, (tr_idx, val_idx) in enumerate(kf.split(Xtr), start=1):
        Xt, Xv = Xtr.iloc[tr_idx], Xtr.iloc[val_idx]
        yt, yv = ytr.iloc[tr_idx], ytr.iloc[val_idx]
        m = model_in.__class__(**model_in.get_params())
        if fit_params: m.fit(Xt, yt, **fit_params, eval_set=[(Xv,yv)])
        else:          m.fit(Xt, yt)
        pred_val = m.predict(Xv)
        oof_pred[val_idx] = pred_val
        test_preds.append(m.predict(Xte))
    test_pred = np.mean(np.vstack(test_preds), axis=0)
    rmse_oof = _rmse(ytr, oof_pred)
    return oof_pred, test_pred, rmse_oof



def build_base_models(seed=42):
    lgb_params = dict(
        n_estimators=3000, learning_rate=0.03, num_leaves=95, min_child_samples=30,
        feature_fraction=0.85, subsample=0.8, subsample_freq=1, lambda_l2=0.5,
        random_state=seed, verbose=-1
    )
    xgb_params = dict(
        n_estimators=2500, learning_rate=0.03, max_depth=8, subsample=0.8,
        colsample_bytree=0.9, reg_lambda=1.0, random_state=seed, tree_method="hist"
    )
    cat_params = dict(
        iterations=2500, learning_rate=0.03, depth=8, random_seed=seed,
        loss_function="RMSE", verbose=0
    )

    return [
        ("LGBM", LGBMRegressor(**lgb_params)),
        ("XGB",  XGBRegressor(**xgb_params)),
        ("CAT",  CatBoostRegressor(**cat_params))
    ]


def run_stacking(results, n_folds=5, seed=42, verbose=True):
    X_train = results['X_train_scaled'].copy()
    X_test  = _align_test_columns(X_train, results['X_test_scaled'].copy())
    y_train = results['y_train'].copy()
    y_test  = results['y_test'].copy()

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    base_models = build_base_models(seed=seed)

    # هیچ شرطی ندارد؛ CatBoost همیشه هست
    fit_opts = {
        "LGBM": dict(eval_metric="rmse", callbacks=[]),
        "XGB":  dict(verbose=False),
        "CAT":  dict(use_best_model=True)
    }

    oof_list, test_list, oof_rmses = [], [], {}
    for name, mdl in base_models:
        oof_pred, test_pred, rmse_oof = oof_model(
            mdl, X_train, y_train, X_test, kf,
            fit_params=fit_opts.get(name), name=name
        )
        oof_list.append(oof_pred)
        test_list.append(test_pred)
        oof_rmses[name] = rmse_oof
        if verbose:
            print(f"{name} OOF RMSE(log): {rmse_oof:.4f}")

    Z_train = np.vstack(oof_list).T
    Z_test  = np.vstack(test_list).T

    meta = RidgeCV(alphas=[0.01, 0.1, 1.0, 3.0, 10.0], cv=5)
    meta.fit(Z_train, y_train)
    y_pred_meta = meta.predict(Z_test)

    rmse  = _rmse(y_test, y_pred_meta)
    mae   = mean_absolute_error(y_test, y_pred_meta)
    r2    = r2_score(y_test, y_pred_meta)
    mape  = mean_absolute_percentage_error(y_test, y_pred_meta) * 100

    if verbose:
        print("\nStacking (LGBM + XGB + CAT) → Meta: RidgeCV")
        print(f"RMSE : {rmse:.4f}\nMAE  : {mae:.4f}\nR²   : {r2:.4f}\nMAPE : {mape:.2f}%")
        try:
            print("Meta Coefs:", meta.coef_)
        except Exception:
            pass

    return {"meta": meta, "metrics": dict(rmse=rmse, mae=mae, r2=r2, mape=mape)}

# =========================
# Recommender
# =========================
from sentence_transformers import SentenceTransformer

def build_recommender(df_all, model_name="sentence-transformers/all-mpnet-base-v2"):
    from ast import literal_eval
    d = df_all.copy()
    d.fillna({'overview':'','tagline':'','Keywords':'[]','genres':'[]','cast':'[]','crew':'[]'}, inplace=True)
    def _safe_list(x):
        if isinstance(x, list): return x
        if isinstance(x, str):
            try: return literal_eval(x)
            except: return []
        return []
    def _set_names(x):
        out=set()
        for q in _safe_list(x):
            if isinstance(q, dict) and q.get("name"):
                out.add(str(q["name"]).strip().lower())
        return out
    def _cast_names(x, top_k=10):
        items=_safe_list(x)
        if items and isinstance(items[0], dict) and 'order' in items[0]:
            items=sorted(items, key=lambda d: d.get('order', 10**9))
        out=[]
        for q in items:
            n=q.get("name")
            if n: out.append(str(n).strip().lower())
        return set(out[:top_k])
    def _dir_names(x):
        out=set()
        for q in _safe_list(x):
            if isinstance(q, dict) and "director" in str(q.get("job","")).lower() and q.get("name"):
                out.add(str(q["name"]).strip().lower())
        return out
    def _join_words(s): return " ".join(sorted(list(s))).replace(" ","_")

    d["combined_text_rec"] = (
        d["overview"].astype(str) + " " + d.get("tagline","").astype(str) + " " +
        d["genres"].apply(_set_names).apply(_join_words) + " " +
        d["Keywords"].apply(_set_names).apply(_join_words) + " " +
        d["cast"].apply(_cast_names).apply(_join_words) + " " +
        d["crew"].apply(_dir_names).apply(_join_words)
    ).str.strip()

    model = SentenceTransformer(model_name)
    E = model.encode(d["combined_text_rec"].tolist(), batch_size=64, convert_to_numpy=True, normalize_embeddings=True)
    titles_full = d['title'].fillna(d['original_title']).astype(str).tolist()
    years = pd.to_datetime(d["release_date"], errors="coerce").dt.year.fillna(-1).astype(int).to_numpy()
    title_map = {}
    def _norm_title(t):
        t = re.sub(r'\s*\(\d{4}\)\s*$','',str(t)); t = re.sub(r'\s+',' ',t).strip().lower(); return t
    for i,t in enumerate(titles_full):
        n=_norm_title(t); title_map.setdefault(n, []).append(i)
    return {"df": d, "model": model, "emb": E, "titles_full": titles_full,
            "title_map": title_map, "years": years,
            "genres": d["genres"].apply(_set_names).tolist(),
            "keywords": d["Keywords"].apply(_set_names).tolist(),
            "cast": d["cast"].apply(_cast_names).tolist(),
            "dirs": d["crew"].apply(_dir_names).tolist()}

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

def recommend_by_title(title, rec, top_k=5,
                       w_sbert=0.45, w_kw=0.30, w_dir=0.15, w_cast=0.10,
                       force_same_genre=False, min_shared_keywords=0,
                       prefer_genres=("drama","mystery","thriller"),
                       year_boost=0.02, year_window=7,
                       title_match_min_sim=0.45,
                       franchise_tokens=('batman','joker','dc'),
                       franchise_penalty=0.05):
    import numpy as np
    def _norm_title_simple(t):
        t = re.sub(r'\s*\(\d{4}\)\s*$','',str(t)); t = re.sub(r'\s+',' ',t).strip().lower(); return t
    def _jac(a,b):
        u=(a|b); return len(a&b)/len(u) if u else 0.0
    df=rec["df"]; E=rec["emb"]; titles=rec["titles_full"]; years=rec["years"]
    G=rec["genres"]; K=rec["keywords"]; C=rec["cast"]; D=rec["dirs"]
    model=rec["model"]; tmap=rec.get("title_map",{})
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
    score = w_sbert*sims_sbert + w_kw*sims_kw + w_dir*sims_dir + w_cast*sims_cast
    pg=set([g.strip().lower() for g in (prefer_genres or [])])
    if pg: score += 0.05*np.array([len(pg & G[j]) for j in range(len(df))])
    mask_valid = np.ones(len(df), dtype=bool)
    if force_same_genre: mask_valid &= np.array([len(G[idx] & G[j]) > 0 for j in range(len(df))])
    if min_shared_keywords > 0: mask_valid &= np.array([len(K[idx] & K[j]) >= min_shared_keywords for j in range(len(df))])
    y0 = years[idx]
    if y0 != -1 and year_window > 0:
        delta = np.abs(years - y0)
        score += year_boost * np.clip((year_window - delta) / year_window, 0, 1)
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
    mask = ~all_recs['title'].str.lower().str.contains('|'.join(franchise_tokens))
    return all_recs[mask].head(top_k).reset_index(drop=True)

def recommend_hybrid(title, rec, top_k=5, nonfr_k=5,
                     franchise_tokens=('batman','joker','gotham','arkham','dark knight'), **kwargs):
    nonfr = recommend_by_title_nonfranchise(title, rec, top_k=nonfr_k,
                                            franchise_tokens=franchise_tokens, **kwargs)
    has_fr = nonfr['title'].str.lower().str.contains('|'.join(franchise_tokens)).any()
    if not has_fr:
        full = recommend_by_title(title, rec, top_k=10,
                                  franchise_tokens=franchise_tokens,
                                  franchise_penalty=0.01, **kwargs)
        for _, row in full.iterrows():
            t = str(row['title']).lower()
            if any(tok in t for tok in franchise_tokens):
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

def patch_title_index(rec):
    titles_full = rec["titles_full"]; model = rec["model"]
    def _norm_title(t):
        t = re.sub(r'\s*\(\d{4}\)\s*$','',str(t)); t = re.sub(r'\s+',' ',t).strip().lower(); return t
    titles_norm=[]; title_map={}
    for i,t in enumerate(titles_full):
        n=_norm_title(t); titles_norm.append(n); title_map.setdefault(n, []).append(i)
    rec["titles_norm"]=titles_norm
    rec["title_map"]=title_map
    rec["title_emb"]=model.encode(titles_full, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)
    return rec

def search_title_candidates(rec, query, topn=10):
    df = rec["df"]; titles = rec["titles_full"]; title_emb = rec.get("title_emb"); model = rec["model"]
    q = str(query).strip()
    mask = df['title'].fillna('').str.contains(q, case=False, regex=False) | df['original_title'].fillna('').str.contains(q, case=False, regex=False)
    hits = df.loc[mask, ['title','original_title','release_date']].head(topn)
    if len(hits) > 0: return hits
    if title_emb is None:
        rec["title_emb"] = model.encode(titles, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)
        title_emb = rec["title_emb"]
    qv = model.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0]
    sims = title_emb @ qv
    order = np.argsort(-sims)[:topn]
    out = [(titles[i], df.iloc[i]['original_title'], df.iloc[i]['release_date']) for i in order]
    return pd.DataFrame(out, columns=["title","original_title","release_date"])


def run_base_model(csv_path):
    df_train = pd.read_csv(csv_path)
    results = prepare_and_train(df_train)
    return results




##############################################################################
# # Modelleme
##############################################################################

results = run_base_model('D:/Bootcamp/Final project/2/tmdb-box-office-prediction/train.csv')
stack = run_stacking(results, n_folds=5, seed=42, verbose=True)


df_all = results['df_all'].copy()
rec = build_recommender(df_all)
rec = patch_title_index(rec)
print(recommend_by_title_nonfranchise("the dark knight", rec, top_k=5))
print(recommend_hybrid("the dark knight", rec, top_k=5))
print(recommend_by_query("the dark knight", rec, top_k=5))

##############################################################################
# # Figürler
##############################################################################


import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

try:
    from lightgbm import LGBMRegressor
    MODEL_NAME = "LightGBM"
    def make_model():
        return LGBMRegressor(
            n_estimators=1200, learning_rate=0.05, num_leaves=63,
            min_child_samples=30, subsample=0.8, colsample_bytree=0.9,
            random_state=42, verbose=-1
        )
except Exception:
    from sklearn.ensemble import RandomForestRegressor
    MODEL_NAME = "RandomForest"
    def make_model():
        return RandomForestRegressor(
            n_estimators=400, random_state=42, n_jobs=-1
        )

def main(csv_path: str):
    df = pd.read_csv(csv_path)

    target = "revenue"
    if target not in df.columns:
        raise ValueError(f"'{target}' column not found in CSV.")

    # Numeric features only (no leakage from target)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != target]
    if len(num_cols) < 2:
        raise ValueError("Need at least two numeric feature columns.")

    X = df[num_cols].copy().fillna(df[num_cols].median(numeric_only=True))
    y = np.log1p(df[target].astype(float).values)  # log-target

    # Split + scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Fit
    model = make_model()
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)

    # Metrics
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = float(mean_absolute_error(y_test, y_pred))
    r2   = float(r2_score(y_test, y_pred))
    print(f"{MODEL_NAME} — metrics on log-revenue")
    print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")

    outdir = "figures"; os.makedirs(outdir, exist_ok=True)

    # 1) Residuals
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=40)
    ax.set_title("Residuals (log-revenue)")
    ax.set_xlabel("Residual"); ax.set_ylabel("Count")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "residuals.png"), dpi=150)

    # 2) Pred vs Actual
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_test, y_pred, s=8, alpha=0.6)
    mn, mx = float(min(y_test.min(), y_pred.min())), float(max(y_test.max(), y_pred.max()))
    ax.plot([mn, mx], [mn, mx], "--", linewidth=1)
    ax.set_xlabel("Actual (log)"); ax.set_ylabel("Predicted (log)")
    ax.set_title("Actual vs Predicted")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "pred_vs_actual.png"), dpi=150)

    # 3) Feature importance (if available)
    try:
        importances = model.feature_importances_
        order = np.argsort(importances)[::-1][:25]
        feat_names = [num_cols[i] for i in order]
        imp_vals   = importances[order]
        h = max(4, 0.3 * len(feat_names))
        fig, ax = plt.subplots(figsize=(8, h))
        ax.barh(feat_names[::-1], imp_vals[::-1])
        ax.set_title("Feature Importance")
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "feature_importance.png"), dpi=150)
    except Exception:
        pass

    # 4) Learning curve — CV (Train/CV RMSE)
    sizes, tr_scores, va_scores = learning_curve(
        make_model(), X_train_sc, y_train,
        train_sizes=np.linspace(0.1, 1.0, 6),
        cv=5, scoring="neg_root_mean_squared_error",
        shuffle=True, random_state=42, n_jobs=-1
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, -tr_scores.mean(axis=1), marker="o", label="Train RMSE (CV)")
    ax.plot(sizes, -va_scores.mean(axis=1), marker="s", label="Validation RMSE (CV)")
    ax.set_xlabel("Training size"); ax.set_ylabel("RMSE"); ax.legend()
    ax.set_title("Learning Curve (CV)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "learning_curve_cv.png"), dpi=150)

    # 5) Learning curve — hold-out (Train/Test RMSE)
    rng = np.random.RandomState(42)
    fracs = np.linspace(0.1, 1.0, 6)
    tr_rmse, te_rmse, n_pts = [], [], []
    for f in fracs:
        n = max(50, int(f * len(X_train_sc)))
        idx = rng.choice(len(X_train_sc), n, replace=False)
        m = make_model()
        m.fit(X_train_sc[idx], y_train[idx])
        tr_rmse.append(np.sqrt(mean_squared_error(y_train[idx], m.predict(X_train_sc[idx]))))
        te_rmse.append(np.sqrt(mean_squared_error(y_test, m.predict(X_test_sc))))
        n_pts.append(n)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_pts, tr_rmse, marker="o", label="Train RMSE")
    ax.plot(n_pts, te_rmse, marker="s", label="Test RMSE")
    ax.set_xlabel("Training size"); ax.set_ylabel("RMSE"); ax.legend()
    ax.set_title("Learning Curve (hold-out)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "learning_curve_holdout.png"), dpi=150)

    # 6) SHAP (optional for tree models)
    try:
        import shap
        X_test_df = pd.DataFrame(X_test_sc, columns=num_cols)
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_test_df.iloc[:min(300, len(X_test_df))])
        plt.figure(figsize=(8, 5))
        shap.summary_plot(sv, X_test_df, show=False)
        plt.tight_layout(); plt.savefig(os.path.join(outdir, "shap_summary.png"), dpi=150)

        # dependence on top feature
        top_feat = num_cols[int(np.argmax(model.feature_importances_))]
        plt.figure(figsize=(8, 5))
        shap.dependence_plot(top_feat, sv, X_test_df, show=False)
        plt.tight_layout(); plt.savefig(os.path.join(outdir, "shap_dependence.png"), dpi=150)
    except Exception as e:
        print(f"[SHAP skipped] {e}")

    # Show windows (optional)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quick_figs_offline.py <path_to_csv>")
        sys.exit(1)
    main(sys.argv[1])

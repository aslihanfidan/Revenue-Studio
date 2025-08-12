
with reco_tab:
    st.subheader("🎬 Öneri Sistemi")

    if train_df_raw.empty:
        st.info("Önce sol menüden train.csv yükleyin. (Tercihen: title / overview / genres sütunları)")
    else:
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

        # --- Metin birleştirme (overview + genres) ---
        def _combined_text(df, title_col, overview_col, genre_col):
            import pandas as pd, ast
            def _genres_to_words(x):
                if pd.isna(x): return ""
                s = str(x)
                try:
                    obj = ast.literal_eval(s)
                    if isinstance(obj, list):
                        names = []
                        for it in obj:
                            if isinstance(it, dict) and it.get("name"):
                                names.append(str(it["name"]))
                            elif isinstance(it, str):
                                names.append(it)
                        return " ".join(names)
                except Exception:
                    pass
                return s  # düz metin ise aynen ekle
            parts = []
            if overview_col and overview_col in df.columns:
                parts.append(df[overview_col].astype(str))
            if genre_col and genre_col in df.columns:
                parts.append(df[genre_col].apply(_genres_to_words).astype(str))
            if not parts:
                return pd.Series([""] * len(df))
            s = pd.Series([""] * len(df))
            for p in parts:
                s = s.str.cat(p.fillna(""), sep=" ")
            return s.str.replace(r"\s+", " ", regex=True).str.strip()

        title_col    = _pick_col(train_df_raw, ["title","original_title","movie_title","Title"])
        overview_col = _pick_col(train_df_raw, ["overview","description","plot","tagline"])
        genre_col    = _pick_col(train_df_raw, ["genres","genre","main_genre"])

        if not title_col:
            st.error("Film başlığı sütunu bulunamadı (title / original_title vb.).")
            st.stop()
        if not (overview_col or genre_col):
            st.error("overview/description veya genres/genre sütunlarından en az biri gerekli.")
            st.stop()

        # --- Vektörleştirme (cache) ---
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        @st.cache_resource(show_spinner=False)
        def _build_vector_space(df_text: pd.Series):
            vec = TfidfVectorizer(stop_words="english", max_features=100_000, ngram_range=(1,2))
            X = vec.fit_transform(df_text.fillna(""))
            return vec, X

        df_rec = train_df_raw[[title_col] + [c for c in [overview_col, genre_col] if c]].copy()
        df_rec["_combined_text_"] = _combined_text(df_rec, title_col, overview_col, genre_col)
        vec, X = _build_vector_space(df_rec["_combined_text_"])

        # --- UI ---
        st.caption("Aşağıdan bir film seçerek veya serbest sorgu yazarak benzer film önerileri alabilirsiniz.")
        colA, colB = st.columns([2, 1])
        with colA:
            titles = (
                df_rec[title_col]
                .astype(str).dropna().drop_duplicates().sort_values()
                .tolist()
            )
            film_sec = st.selectbox("Film seç (title-based / hybrid için):", titles, index=0)
        with colB:
            top_k = st.slider("Öneri sayısı", 3, 10, 5)

        query_text = st.text_input("Serbest metin (query-based için):", value="dark gritty crime thriller")
        tokens_default = "batman,joker,gotham,arkham,dark knight"
        franchise_tokens_in = st.text_input("Franchise filtreleme anahtarları (virgüllü):", value=tokens_default)
        franchise_tokens = tuple([t.strip().lower() for t in franchise_tokens_in.split(",") if t.strip()])

        t1, t2, t3, t4 = st.tabs(["Title-based", "Non-franchise", "Query-based", "Hybrid"])

        # --- Yardımcılar ---
        def _get_index_for_title(df, title_col, title_str):
            tlist = df[title_col].astype(str).str.lower().tolist()
            tlow = str(title_str).lower()
            try:
                return tlist.index(tlow)
            except ValueError:
                matches = [i for i, t in enumerate(tlist) if t == tlow]
                return matches[0] if matches else 0

        def _mask_nonfranchise(title_series, tokens):
            t = title_series.astype(str).str.lower()
            if not tokens:
                return pd.Series([True]*len(t), index=title_series.index)
            patt = "|".join([re.escape(tok) for tok in tokens])
            return ~t.str.contains(patt, na=False)

        import re
        # --- 1) Title-based ---
        with t1:
            try:
                idx = _get_index_for_title(df_rec, title_col, film_sec)
                sims = cosine_similarity(X[idx], X).ravel()
                sims[idx] = -1.0
                order = np.argsort(-sims)[:top_k]
                out = df_rec.iloc[order][[title_col]].copy()
                out["similarity"] = sims[order]
                st.dataframe(out.reset_index(drop=True), use_container_width=True)
            except Exception as e:
                st.warning(f"Başlık bazlı öneri oluşturulamadı: {e}")

        # --- 2) Non-franchise Title-based ---
        with t2:
            try:
                idx = _get_index_for_title(df_rec, title_col, film_sec)
                sims = cosine_similarity(X[idx], X).ravel()
                sims[idx] = -1.0
                # franchise filtre
                mask_nonfr = _mask_nonfranchise(df_rec[title_col], franchise_tokens).to_numpy()
                sims[~mask_nonfr] = -1.0
                order = np.argsort(-sims)[:top_k]
                out = df_rec.iloc[order][[title_col]].copy()
                out["similarity"] = sims[order]
                st.dataframe(out.reset_index(drop=True), use_container_width=True)
            except Exception as e:
                st.warning(f"Non-franchise öneri oluşturulamadı: {e}")

        # --- 3) Query-based ---
        with t3:
            try:
                if not query_text.strip():
                    st.info("Bir sorgu yazın (ör. 'space survival thriller').")
                else:
                    Xq = vec.transform([query_text])
                    sims = cosine_similarity(Xq, X).ravel()
                    order = np.argsort(-sims)[:top_k]
                    out = df_rec.iloc[order][[title_col]].copy()
                    out["similarity"] = sims[order]
                    st.dataframe(out.reset_index(drop=True), use_container_width=True)
            except Exception as e:
                st.warning(f"Query-based öneri oluşturulamadı: {e}")

        # --- 4) Hybrid (önce non-franchise, yoksa 1 franchise ekle) ---
        with t4:
            try:
                idx = _get_index_for_title(df_rec, title_col, film_sec)
                sims_full = cosine_similarity(X[idx], X).ravel()
                sims_full[idx] = -1.0

                # non-franchise ilk liste
                mask_nonfr = _mask_nonfranchise(df_rec[title_col], franchise_tokens).to_numpy()
                sims_nonfr = sims_full.copy()
                sims_nonfr[~mask_nonfr] = -1.0
                order_nonfr = np.argsort(-sims_nonfr)

                picks = []
                for j in order_nonfr:
                    if sims_nonfr[j] <= -1.0: break
                    picks.append(j)
                    if len(picks) == top_k: break

                # eğer hiç franchise içermiyorsa, full listeden bir tane franchise ekle
                contains_franchise = False
                if picks:
                    titles_lower = df_rec.iloc[picks][title_col].astype(str).str.lower().tolist()
                    if franchise_tokens:
                        patt = "|".join([re.escape(tok) for tok in franchise_tokens])
                        contains_franchise = any(re.search(patt, t) for t in titles_lower)

                if (not contains_franchise) and len(picks) < top_k and franchise_tokens:
                    patt = "|".join([re.escape(tok) for tok in franchise_tokens])
                    titles_lower_full = df_rec[title_col].astype(str).str.lower().tolist()
                    # full sıralamada ilk franchise'ı bul
                    for j in np.argsort(-sims_full):
                        if sims_full[j] <= -1.0: break
                        if re.search(patt, titles_lower_full[j]) and j not in picks:
                            picks.append(j)
                            break

                picks = picks[:top_k]
                out = df_rec.iloc[picks][[title_col]].copy()
                # hangi benzerlikten geldiğini görmek için full similarity yazıyoruz
                out["similarity"] = sims_full[pd.Index(df_rec.index).get_indexer(picks)]
                st.dataframe(out.reset_index(drop=True), use_container_width=True)
            except Exception as e:
                st.warning(f"Hybrid öneri oluşturulamadı: {e}")
                

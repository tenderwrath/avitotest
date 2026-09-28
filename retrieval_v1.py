"""Локальная кандидатогенерация услуг: лексический поиск + обучение на парах.

Индексы строятся по доступным текстам корпуса. История используется только из
обучающей части: ни query_id, ни тестовые ответы в признаках не используются.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
import argparse, json, re, time, pickle, gc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
    TfidfTransformer,
)
from sklearn.preprocessing import normalize
from nltk.stem.snowball import RussianStemmer
from catboost import CatBoostClassifier
from validate_answer import validate_answer

SEED = 20260927
STEM = RussianStemmer()


@lru_cache(maxsize=350000)
def stem(w):
    return STEM.stem(w) if re.search("[а-я]", w) else w


def clean(text):
    return " ".join(
        re.findall(
            r"[а-яa-z0-9]+",
            str(text or "").lower().replace("ё", "е").replace("\\n", " "),
        )
    )


def tokenize(text):
    return " ".join(stem(w) for w in clean(text).split())


def topk(a, k):
    k = min(k, len(a))
    z = np.argpartition(a, len(a) - k)[-k:]
    return z[np.lexsort((z, -a[z]))]


def bm25(x, idf, b=0.65):
    """Сатурация частоты термина и нормализация длины документа BM25."""
    x = x.astype(np.float32).tocsr()
    lens = np.asarray(x.sum(1)).ravel()
    denom = 1.3 * (1 - b + b * lens / max(lens.mean(), 1))
    x.data = x.data * 2.3 / (x.data + np.repeat(denom, np.diff(x.indptr)))
    x.data *= idf[x.indices]
    return x


def array_scores(q, x):
    return np.asarray((q @ x.T).toarray()).ravel()


class Retriever:
    def __init__(self, items, history):
        self.items = items.reset_index(drop=True)
        self.n = len(items)
        self.ids = self.items.item_id.to_numpy()
        self.idmap = {s: k for k, s in enumerate(self.ids)}
        self.history = history
        print("Tokenize corpus", flush=True)
        self.titles = self.items.item_title_raw.fillna("").map(clean).tolist()
        title = self.items.item_title_raw.fillna("").map(tokenize)
        desc = (
            self.items.item_description_raw.fillna("").str.slice(0, 7000).map(tokenize)
        )
        params = self.items.item_infm_params_text.fillna("").map(tokenize)
        # Общий словарь: полные названия, параметры и первые 7000 символов описания.
        self.word = CountVectorizer(
            tokenizer=str.split,
            preprocessor=None,
            token_pattern=None,
            min_df=2,
            max_features=220000,
            dtype=np.float32,
        )
        x = self.word.fit_transform(title + " " + desc + " " + params)
        df = np.asarray((x > 0).sum(0)).ravel()
        self.idf = np.log1p((self.n - df + 0.5) / (df + 0.5)).astype("float32")
        del x
        self.bt = bm25(self.word.transform(title), self.idf, b=0.4)
        self.bd = bm25(self.word.transform(desc), self.idf, b=0.7)
        self.bp = bm25(self.word.transform(params), self.idf, b=0.4)
        self.tt = normalize(self.word.transform(title).multiply(self.idf)).tocsr()
        print("Character index", flush=True)
        self.char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=3,
            max_features=180000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.ct = self.char.fit_transform(self.titles).tocsc()
        for attr in ["bt", "bd", "bp", "tt"]:
            setattr(self, attr, getattr(self, attr).tocsc())
        del title, desc, params
        gc.collect()
        self.build_history(history)
        print("Index ready", flush=True)

    def build_history(self, history):
        """Переобучить признаки поведения на переданной разрешённой истории.

        Вызывается сначала на истории без валидационных пар, затем на всём
        train перед отправкой. Текстовые индексы корпуса при этом неизменны.
        """
        print("Training query graph", len(history), flush=True)
        self.history = history
        h = history.copy()
        h["norm"] = h.search_query.map(clean)
        self.hq = np.array(sorted(h.norm.unique()))
        qi = {x: i for i, x in enumerate(self.hq)}
        r = h.norm.map(qi).to_numpy()
        self.qword = TfidfVectorizer(
            tokenizer=str.split,
            preprocessor=None,
            token_pattern=None,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.qw = self.qword.fit_transform([tokenize(s) for s in self.hq]).tocsc()
        self.qchar = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=180000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.qc = self.qchar.fit_transform(self.hq).tocsc()
        self.micro_values = np.unique(
            np.r_[self.items.item_microcat_id.to_numpy(), h.item_microcat_id.to_numpy()]
        )
        mi = {x: i for i, x in enumerate(self.micro_values)}
        c = h.item_microcat_id.map(mi).to_numpy()
        self.qmicro = normalize(
            sparse.coo_matrix(
                (np.ones(len(h), dtype="float32"), (r, c)), shape=(len(qi), len(mi))
            ).tocsr(),
            norm="l1",
        )
        self.imicro = self.items.item_microcat_id.map(mi).to_numpy()
        # Средние TF-IDF названия выбранных объявлений дают расширение запроса.
        ut = h[["item_id", "item_title_raw"]].drop_duplicates("item_id")
        ti = {v: k for k, v in enumerate(ut.item_id)}
        qt = normalize(
            sparse.coo_matrix(
                (np.ones(len(h), dtype="float32"), (r, h.item_id.map(ti))),
                shape=(len(qi), len(ti)),
            ).tocsr(),
            norm="l1",
        )
        tx = normalize(
            self.word.transform(ut.item_title_raw.fillna("").map(tokenize)).multiply(
                self.idf
            )
        )
        self.exp = (qt @ tx).tocsr()
        # Исторические выборы конкретных объявлений — только из train.
        hx = h[h.item_id.isin(self.idmap)]
        self.qitem = sparse.coo_matrix(
            (
                np.ones(len(hx), dtype="float32"),
                (hx.norm.map(qi), hx.item_id.map(self.idmap)),
            ),
            shape=(len(qi), self.n),
        ).tocsr()
        self.pop = (
            self.items.item_id.map(h.item_id.value_counts())
            .fillna(0)
            .to_numpy(dtype="float32")
        )
        loc = h.groupby(["search_location_id", "item_location_id"]).size()
        self.locpairs = loc.to_dict()
        self.loctotal = h.search_location_id.value_counts().to_dict()
        self.iloc = self.items.item_location_id.to_numpy()
        coords = self.items[
            ["item_location_id", "item_latitude", "item_longitude"]
        ].copy()
        coords[["item_latitude", "item_longitude"]] = coords[
            ["item_latitude", "item_longitude"]
        ].astype(float)
        self.centers = (
            coords.groupby("item_location_id")[["item_latitude", "item_longitude"]]
            .median()
            .to_dict("index")
        )
        self.lat = np.nan_to_num(coords.item_latitude.to_numpy(), nan=0)
        self.lon = np.nan_to_num(coords.item_longitude.to_numpy(), nan=0)
        self.price = np.nan_to_num(self.items.item_price.to_numpy(dtype=float), nan=0)
        self.rating = self.items.item_rating.fillna(0).to_numpy(dtype=float)
        self.reviews = self.items.item_rating_reviews_count.fillna(0).to_numpy(
            dtype=float
        )
        self.cat = self.items.item_category_id.to_numpy()
        self.locations = np.unique(self.iloc)
        self.locindex = np.searchsorted(self.locations, self.iloc)
        self.item_params = (
            self.items.item_infm_params_text.fillna("").map(clean).tolist()
        )
        self.phone = self.items.item_is_phone_hidden.fillna(False).to_numpy(dtype=float)
        self.message = self.items.item_is_message_forbidden.fillna(False).to_numpy(
            dtype=float
        )
        self.title_tokens = [
            set(s.split()) for s in self.items.item_title_raw.fillna("").map(tokenize)
        ]

    def retrieve(self, row, feature=True):
        """Вернуть объединение кандидатов и числовые признаки для CatBoost.

        Вход содержит только search_*; query_id не нужен для поиска.
        Поиск по разреженным инвертированным индексам не создаёт матрицу
        размера «все запросы × весь корпус» в оперативной памяти.
        """
        raw = clean(row["search_query"])
        tok = tokenize(raw)
        qw = self.word.transform([tok])
        qw.data[:] = 1
        bt = array_scores(qw, self.bt)
        bd = array_scores(qw, self.bd)
        bp = array_scores(qw, self.bp)
        ch = array_scores(self.char.transform([raw]), self.ct)
        # Ближайшие обучающие запросы: слова + устойчивые к опечаткам n-граммы.
        ns = 0.45 * array_scores(
            self.qword.transform([tok]), self.qw
        ) + 0.55 * array_scores(self.qchar.transform([raw]), self.qc)
        near = topk(ns, 35)
        weights = np.maximum(ns[near], 0) ** 8
        if weights.sum() > 0:
            weights /= weights.sum()
        v = sparse.csr_matrix(
            (weights, (np.zeros(len(near), dtype=int), near)), shape=(1, len(self.hq))
        )
        mp = np.asarray((v @ self.qmicro).toarray()).ravel()[self.imicro]
        ex = (v @ self.exp).tocsr()
        if ex.nnz > 24:
            keep = topk(ex.data, 24)
            ex = sparse.csr_matrix(
                (ex.data[keep], (np.zeros(len(keep), int), ex.indices[keep])),
                shape=ex.shape,
            )
        ex = normalize(ex)
        es = array_scores(ex, self.tt)
        hist = np.asarray((v @ self.qitem).toarray()).ravel()
        loc = int(row["search_location_id"])
        same = (self.iloc == loc).astype("float32")
        center = self.centers.get(loc)
        if center:
            a = np.radians(self.lat)
            b = np.radians(center["item_latitude"])
            hav = (
                np.sin((a - b) / 2) ** 2
                + np.cos(a)
                * np.cos(b)
                * np.sin(np.radians(self.lon - center["item_longitude"]) / 2) ** 2
            )
            dist = 6371 * 2 * np.arcsin(np.sqrt(np.clip(hav, 0, 1)))
        else:
            dist = np.full(self.n, 1000.0)
        dist = np.where(same, 0, dist)
        lp = np.array(
            [
                self.locpairs.get((loc, l), 0) / max(self.loctotal.get(loc, 0), 1)
                for l in self.locations
            ]
        )[self.locindex]
        geo = 1.0 * same + 0.55 * np.exp(-dist / 35) + 0.3 * np.sqrt(lp)
        lex = 1.5 * bt + 0.65 * bd + 0.15 * bp
        # Союз нескольких выдач повышает полноту до финального отбора 50.
        lexical = lex * (0.45 + geo) + 10 * ch * (0.4 + geo)
        hybrid = lexical + 8 * es * (0.4 + geo) + 3 * mp * (0.3 + geo)
        expand = (10 * es + 4 * mp + 3 * ch) * (0.35 + geo) + 0.1 * lex
        candidates = np.unique(
            np.concatenate(
                [
                    topk(hybrid, 300),
                    topk(lex, 130),
                    topk(ch * (0.35 + geo), 100),
                    topk(expand, 160),
                    topk(hist * (0.3 + geo), 50),
                ]
            )
        )
        if not feature:
            return candidates
        c = candidates
        qt = set(tok.split())
        nt = max(len(qt), 1)
        overlap = np.array(
            [len(qt & self.title_tokens[j]) / nt for j in c], dtype="float32"
        )
        exact = np.array([float(raw in self.titles[j]) for j in c])
        fclean = clean(row.get("search_infm_params_text", ""))
        ft = set(tokenize(fclean).split())
        # Параметры используются мягко: строгий фильтр снижает полноту.
        fq = self.word.transform([tokenize(fclean)])
        fq.data[:] = 1
        fs = array_scores(fq, self.bp) if fclean else np.zeros(self.n)
        feats = np.column_stack(
            [
                bt[c],
                bd[c],
                bp[c],
                ch[c],
                es[c],
                mp[c],
                np.log1p(hist[c]),
                same[c],
                np.log1p(dist[c]),
                lp[c],
                np.log1p(self.pop[c]),
                np.log1p(np.maximum(self.price[c], 0)),
                self.rating[c],
                np.log1p(self.reviews[c]),
                self.phone[c],
                self.message[c],
                (self.cat[c] == row.get("search_category", 114)),
                fs[c],
                overlap,
                exact,
                hybrid[c],
                lexical[c],
                expand[c],
                np.full(len(c), ns[near[0]]),
                np.full(len(c), len(qt)),
                np.full(len(c), len(raw)),
                np.full(len(c), len(ft)),
            ]
        ).astype("float32")
        return c, feats, hybrid[c], lexical[c]


FEATURES = [
    "title_bm25",
    "description_bm25",
    "params_bm25",
    "title_char_cosine",
    "expanded_title_cosine",
    "microcat_probability",
    "history_log",
    "same_location",
    "distance_log_km",
    "location_probability",
    "popularity_log",
    "price_log",
    "rating",
    "reviews_log",
    "phone_hidden",
    "message_forbidden",
    "category_equal",
    "filters_bm25",
    "query_title_coverage",
    "query_in_title",
    "hybrid_score",
    "lexical_score",
    "expansion_score",
    "nearest_train_query_similarity",
    "query_words",
    "query_characters",
    "filter_words",
]


def load_data(data):
    items = pd.read_parquet(data / "benchmark_items.parquet")
    cols = [
        "search_query",
        "search_location_id",
        "search_is_delivery_search",
        "search_infm_params_text",
        "search_category",
        "item_id",
        "item_title_raw",
        "item_microcat_id",
        "item_location_id",
    ]
    h = pd.read_parquet(data / "train.parquet", columns=cols).fillna(
        {"search_query": "", "search_infm_params_text": "", "item_title_raw": ""}
    )
    q = pd.read_parquet(data / "benchmark_queries.parquet").fillna(
        {"search_query": "", "search_infm_params_text": ""}
    )
    return items, h, q


def make_split(h, items):
    """Все строки одного текста запроса остаются в одной части, без утечки.
    Качество измеряется по известным положительным парам в реальном корпусе.
    Это cold-query validation; она не является оценкой скрытого leaderboard.
    """
    hc = h[h.item_id.isin(set(items.item_id))].copy()
    hc["norm"] = hc.search_query.map(clean)
    # Редкие запросы ближе к бенчмарку; отбор не зависит от релевантных item_id.
    counts = h.search_query.map(clean).value_counts()
    names = np.array(sorted(hc.norm.unique()))
    rng = np.random.default_rng(SEED)
    probs = np.array([1 / (1 + np.log1p(counts[x])) for x in names])
    probs /= probs.sum()
    selected = rng.choice(names, size=min(4000, len(names)), replace=False, p=probs)
    split = {
        s: ("fit" if k < 2400 else "dev" if k < 3200 else "test")
        for k, s in enumerate(selected)
    }
    held = set(selected)
    base = h[~h.search_query.map(clean).isin(held)].copy()
    keys = [
        "search_query",
        "search_location_id",
        "search_is_delivery_search",
        "search_infm_params_text",
        "search_category",
    ]
    records = []
    for s, part in split.items():
        groups = (
            hc[hc.norm == s]
            .groupby(keys, dropna=False)
            .item_id.agg(lambda v: sorted(set(v)))
        )
        # Один случайный контекст на текст, чтобы частые запросы не доминировали.
        j = int(rng.integers(len(groups)))
        key = groups.index[j]
        row = dict(zip(keys, key))
        row["positives"] = groups.iloc[j]
        row["split"] = part
        records.append(row)
    # У 90% отложенных позитивов убираем всю историю объявления, включая
    # другие запросы. Это приближает проверку к преимущественно новому корпусу.
    positive_ids = sorted({x for r in records for x in r["positives"]})
    cold_ids = set(
        rng.choice(positive_ids, size=int(0.9 * len(positive_ids)), replace=False)
    )
    base = base[~base.item_id.isin(cold_ids)].copy()
    # Независимо перемешиваем роли после взвешенного выбора текстов:
    # порядок weighted sampling не должен менять сложность fit/dev/test.
    order = np.random.default_rng(SEED + 1).permutation(len(records))
    for k, j in enumerate(order):
        records[j]["split"] = "fit" if k < 2400 else "dev" if k < 3200 else "test"
    return base, records


def build_features(ret, records, path):
    """Сохранить признаки и разметку, не подмешивая пропущенные позитивы.

    bounds связывает плоский массив кандидатов с исходными запросами.
    Метрика затем усредняется по запросам, а не по строкам этих массивов.
    Три потока разделяют один индекс в памяти и сохраняют порядок входа.
    """
    allx = []
    ally = []
    cs = []
    bounds = [0]
    recalls = []
    hs = []
    ls = []
    pool = ThreadPoolExecutor(max_workers=3)
    for k, (row, result) in enumerate(zip(records, pool.map(ret.retrieve, records))):
        c, x, hy, le = result
        pos = set(row["positives"])
        y = np.array([ret.ids[j] in pos for j in c], dtype="uint8")
        allx.append(x)
        ally.append(y)
        cs.append(c)
        hs.append(hy)
        ls.append(le)
        bounds.append(bounds[-1] + len(c))
        recalls.append(y.sum() / len(pos))
        if (k + 1) % 100 == 0:
            print(
                "Feature queries",
                k + 1,
                "candidate recall",
                round(float(np.mean(recalls)), 4),
                flush=True,
            )
    np.savez_compressed(
        path,
        x=np.concatenate(allx),
        y=np.concatenate(ally),
        c=np.concatenate(cs),
        bounds=np.array(bounds),
        hy=np.concatenate(hs),
        lex=np.concatenate(ls),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("upload"))
    p.add_argument("--work", type=Path, default=Path("work"))
    p.add_argument(
        "--mode", choices=["prepare", "features", "fit", "predict"], required=True
    )
    p.add_argument("--output", type=Path, default=Path("answer.csv"))
    args = p.parse_args()
    args.work.mkdir(exist_ok=True, parents=True)
    if args.mode == "prepare":
        items, h, q = load_data(args.data)
        base, records = make_split(h, items)
        (args.work / "validation_records.json").write_text(
            json.dumps(records, ensure_ascii=False, default=lambda x: int(x)),
            encoding="utf-8",
        )
        ret = Retriever(items, base)
        with open(args.work / "index.pkl", "wb") as f:
            pickle.dump(ret, f, protocol=5)
        build_features(ret, records, args.work / "features.npz")
    elif args.mode == "features":
        records = json.loads((args.work / "validation_records.json").read_text())
        with open(args.work / "index.pkl", "rb") as f:
            ret = pickle.load(f)
        build_features(ret, records, args.work / "features.npz")
    elif args.mode == "fit":
        records = json.loads((args.work / "validation_records.json").read_text())
        d = np.load(args.work / "features.npz")
        x = d["x"]
        y = d["y"]
        bounds = d["bounds"]
        spl = np.array([r["split"] for r in records])
        rows = np.repeat(np.arange(len(records)), np.diff(bounds))
        tr = spl[rows] == "fit"
        dv = spl[rows] == "dev"
        # Неразмеченные кандидаты являются sampled negatives, а не доказанно нерелевантными.
        model = CatBoostClassifier(
            iterations=900,
            depth=7,
            learning_rate=0.055,
            loss_function="Logloss",
            eval_metric="Logloss",
            random_seed=SEED,
            thread_count=6,
            l2_leaf_reg=8,
            verbose=100,
        )
        model.fit(x[tr], y[tr], eval_set=(x[dv], y[dv]), early_stopping_rounds=100)
        model.save_model(str(args.work / "model_validation.cbm"))
        score = model.predict_proba(x)[:, 1]
        metrics = {}
        lexical_scores, hybrid_scores = d["lex"], d["hy"]
        for part in ["fit", "dev", "test"]:
            vals = {
                "queries": int((spl == part).sum()),
                "candidate_recall": [],
                "lexical_recall50": [],
                "hybrid_recall50": [],
                "model_recall50": [],
            }
            for k in np.flatnonzero(spl == part):
                a, b = bounds[k : k + 2]
                n = len(records[k]["positives"])
                yy = y[a:b]
                vals["candidate_recall"].append(float(yy.sum() / n))
                for field, sc in [
                    ("lexical_recall50", lexical_scores),
                    ("hybrid_recall50", hybrid_scores),
                    ("model_recall50", score),
                ]:
                    vals[field].append(float(yy[topk(sc[a:b], 50)].sum() / n))
            for key in list(vals):
                if key != "queries":
                    vals[key] = float(np.mean(vals[key]))
            metrics[part] = vals
        metrics["iterations"] = model.tree_count_
        metrics["features"] = dict(
            zip(FEATURES, map(float, model.feature_importances_))
        )
        (args.work / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False)
        )
        print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
        # Метрики выше получены ДО финального дообучения. После фиксации числа
        # деревьев используем все 4000 запросов из train для модели отправки.
        final_model = CatBoostClassifier(
            iterations=model.tree_count_,
            depth=7,
            learning_rate=0.055,
            loss_function="Logloss",
            random_seed=SEED,
            thread_count=6,
            l2_leaf_reg=8,
            verbose=100,
        )
        final_model.fit(x, y)
        final_model.save_model(str(args.work / "model.cbm"))
    else:
        items, h, q = load_data(args.data)
        with open(args.work / "index.pkl", "rb") as f:
            ret = pickle.load(f)
        # После выбора параметров возвращаем всю разрешённую обучающую историю.
        ret.build_history(h)
        model = CatBoostClassifier()
        model.load_model(str(args.work / "model.cbm"))
        pred = []
        pool = ThreadPoolExecutor(max_workers=3)
        for k, (c, x, _, _) in enumerate(pool.map(ret.retrieve, q.to_dict("records"))):
            s = model.predict_proba(x, thread_count=1)[:, 1]
            pred.append(" ".join(ret.ids[c[topk(s, 50)]]))
            if (k + 1) % 100 == 0:
                print("Predict", k + 1, flush=True)
        answer = pd.DataFrame({"query_id": q.query_id, "answer": pred})
        answer.to_csv(args.output, index=False, encoding="utf-8")
        validate_answer(args.output, q, items)

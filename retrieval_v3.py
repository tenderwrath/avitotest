"""Третья версия: семантическое сравнение запроса с заголовком и описанием."""
import numpy as np
import json
from scipy import sparse
from sklearn.preprocessing import normalize
from retrieval_v2 import Retriever as RetrieverV1, FEATURES as BASE_FEATURES
from retrieval_v1 import clean,tokenize,topk,array_scores

FEATURES=BASE_FEATURES+['e5_description_cosine','e5_description_geo']

class Retriever(RetrieverV1):
    def attach(self,work,description_work=None):
        from pathlib import Path
        work=Path(work)
        self.item_dense=np.load(work/'item_dense.npy',mmap_mode='r')
        self.item_description_dense=np.load((Path(description_work) if description_work else work)/'item_description_dense.npy',mmap_mode='r')
        assert self.item_description_dense.shape==self.item_dense.shape
        ids=np.load(work/'dense_item_ids.npy',allow_pickle=True)
        assert np.array_equal(ids,self.ids),'Порядок объявлений не совпал с эмбеддингами'
        self.query_dense=np.load(work/'query_dense.npy')
        self.query_dense_map={q:i for i,q in enumerate(json.loads((work/'dense_query_texts.json').read_text()))}
        for name,mat in [('title_presence',self.bt),('description_presence',self.bd),('params_presence',self.bp)]:
            value=mat.copy();value.data[:]=1;setattr(self,name,value)
        counts=self.items.item_location_id.value_counts()
        self.item_location_prior=self.items.item_location_id.map(counts).to_numpy(dtype='float32')/self.n
        self.history_query_map={s:i for i,s in enumerate(self.hq)}

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
        original_candidates = candidates
        qdense = self.query_dense[self.query_dense_map[row["search_query"]]]
        dense_score = self.item_dense @ qdense
        dense_geo = dense_score + .035 * same - .012 * np.log1p(dist / 30)
        # Второй семантический канал считывает начало описания вместе с заголовком.
        # Две выдачи (глобальная и с географией) возвращают объявления, которые
        # могли не попасть в BM25 и в выдачу по одному заголовку.
        desc_score = self.item_description_dense @ qdense
        desc_geo = desc_score + .035 * same - .012 * np.log1p(dist / 30)
        v2_candidates=np.unique(np.concatenate([original_candidates,topk(dense_score,150),topk(dense_geo,250)]))
        candidates = np.unique(np.concatenate([v2_candidates,topk(desc_score,150),topk(desc_geo,250)]))
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
        # Доля IDF-веса слов запроса, найденная в каждом поле.
        wq = qw.copy()
        wq.data = self.idf[wq.indices].copy()
        idf_sum = max(float(wq.data.sum()), 1e-6)
        tcov = array_scores(wq, self.title_presence) / idf_sum
        dcov = array_scores(wq, self.description_presence) / idf_sum
        pcov = array_scores(wq, self.params_presence) / idf_sum
        # Общая частота локации учитывает различие размеров городов в корпусе.
        relative_loc = lp / np.maximum(self.item_location_prior, 1e-5)
        qhist = self.history_query_map.get(raw)
        exact_hist = np.zeros(self.n, dtype='float32') if qhist is None else self.qitem.getrow(qhist).toarray().ravel()
        extra = np.column_stack([
            dense_score[c], dense_geo[c], tcov[c], dcov[c], pcov[c],
            bt[c] / idf_sum, bd[c] / idf_sum, np.log1p(relative_loc[c]),
            np.log1p(exact_hist[c]), np.full(len(c), dense_score.max()),
            dense_score[c] - dense_score.max(),
            np.full(len(c), len(wq.indices) / max(len(qt), 1)),
            (np.maximum(tcov[c], dcov[c]) >= .999).astype('float32'),
            np.full(len(c), float(qhist is not None)),
            np.full(len(c), float(row.get('search_category',114)==0)),
        ]).astype('float32')
        feats = np.column_stack([feats,extra,desc_score[c],desc_geo[c]])
        return c, feats, hybrid[c], lexical[c], np.isin(c, v2_candidates)


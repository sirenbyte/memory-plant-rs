#!/usr/bin/env python3
"""Retrieval micro-benchmark data builder for Memory Plant.

Embeds a curated, LABELED multilingual corpus + queries with
multilingual-e5-small via MLX (the Qwen runtime family, no ORT), and dumps
JSON. A Rust harness (src/bin/bench_retrieval.rs) then loads it into Memory
Plant and computes Recall@k / MRR.

HONEST SCOPE: this measures *retrieval* quality (the layer MP owns) — given a
query, is the relevant passage in the top-k. It is NOT LongMemEval (no
multi-session/temporal-reasoning/answer-generation). ru-heavy, with hard
negatives (same-topic distractors) so the number is meaningful, not trivially 1.
"""
import json
import sys

import mlx.core as mx
from mlx_embeddings.utils import load

MODEL = "intfloat/multilingual-e5-small"

# (id, lang, text). Topical clusters → same-topic passages are hard negatives.
CORPUS = [
    # authors (ru)
    ("auth_tolstoy", "ru", "Роман «Война и мир» написал Лев Толстой."),
    ("auth_dostoevsky", "ru", "Роман «Преступление и наказание» написал Фёдор Достоевский."),
    ("auth_pushkin", "ru", "Роман в стихах «Евгений Онегин» написал Александр Пушкин."),
    ("auth_gogol", "ru", "Поэму «Мёртвые души» написал Николай Гоголь."),
    # capitals (ru + kz)
    ("cap_france", "ru", "Столица Франции — город Париж."),
    ("cap_japan", "ru", "Столица Японии — город Токио."),
    ("cap_kz", "kz", "Қазақстанның астанасы — Астана қаласы."),
    ("cap_germany", "ru", "Столица Германии — город Берлин."),
    # science (ru + en)
    ("sci_mito", "en", "The mitochondria is the powerhouse of the cell."),
    ("sci_water", "ru", "Вода состоит из водорода и кислорода."),
    ("sci_light", "ru", "Скорость света в вакууме примерно триста тысяч километров в секунду."),
    ("sci_photo", "en", "Photosynthesis converts light into chemical energy in plants."),
    # cooking (ru)
    ("cook_borsch", "ru", "Для борща нужны свёкла, капуста, картофель и мясо."),
    ("cook_plov", "ru", "Плов готовят из риса, моркови, лука и мяса."),
    ("cook_blini", "ru", "Блины пекут из муки, молока и яиц."),
    # programming (ru + en)
    ("tech_rust", "ru", "Rust — системный язык программирования без сборщика мусора."),
    ("tech_python", "ru", "Python — интерпретируемый язык с динамической типизацией."),
    ("tech_http", "en", "HTTP is an application protocol that runs on top of TCP."),
    ("tech_gc", "en", "Garbage collection automatically frees unused memory at runtime."),
    # geography (ru)
    ("geo_baikal", "ru", "Байкал — самое глубокое озеро на Земле."),
    ("geo_everest", "ru", "Эверест — самая высокая гора в мире."),
    ("geo_nile", "ru", "Нил — одна из самых длинных рек в мире."),
    # history (ru)
    ("hist_gagarin", "ru", "Первым человеком в космосе стал Юрий Гагарин в 1961 году."),
    ("hist_ussr", "ru", "Советский Союз прекратил существование в 1991 году."),
    ("hist_wwii", "ru", "Вторая мировая война закончилась в 1945 году."),
    # personal facts (the assistant use case) ru
    ("me_job", "ru", "Пользователь работает инженером-программистом."),
    ("me_city", "ru", "Пользователь живёт в городе Алматы."),
    ("me_hobby", "ru", "Пользователь увлекается горными лыжами и шахматами."),
    ("me_lang", "ru", "Пользователь предпочитает язык программирования Rust."),
    # kz cluster
    ("kz_abai", "kz", "Абай Құнанбайұлы — қазақтың ұлы ақыны."),
    ("kz_almaty", "kz", "Алматы — Қазақстанның ең ірі қаласы."),
    # en distractors
    ("en_paris", "en", "Paris is the capital city of France."),
    ("en_rust", "en", "Rust is a systems programming language with no garbage collector."),
]

# (id, lang, query, [relevant_ids]). Paraphrased so it's semantic, not lexical.
QUERIES = [
    ("q1", "ru", "Кто автор романа Война и мир?", ["auth_tolstoy"]),
    ("q2", "ru", "Кто написал Преступление и наказание?", ["auth_dostoevsky"]),
    ("q3", "ru", "Автор Евгения Онегина?", ["auth_pushkin"]),
    ("q4", "ru", "Какой город является столицей Франции?", ["cap_france", "en_paris"]),
    ("q5", "ru", "Назови столицу Японии", ["cap_japan"]),
    ("q6", "ru", "Главный город Казахстана?", ["cap_kz"]),
    ("q7", "ru", "Что является энергетической станцией клетки?", ["sci_mito"]),
    ("q8", "ru", "Из чего состоит вода?", ["sci_water"]),
    ("q9", "ru", "Чему равна скорость света?", ["sci_light"]),
    ("q10", "ru", "Какие ингредиенты нужны для борща?", ["cook_borsch"]),
    ("q11", "ru", "Как готовят плов?", ["cook_plov"]),
    ("q12", "ru", "Какой язык программирования не имеет сборщика мусора?", ["tech_rust", "en_rust"]),
    ("q13", "ru", "Расскажи про язык Python", ["tech_python"]),
    ("q14", "ru", "Самое глубокое озеро?", ["geo_baikal"]),
    ("q15", "ru", "Самая высокая гора на планете?", ["geo_everest"]),
    ("q16", "ru", "Кто первым полетел в космос?", ["hist_gagarin"]),
    ("q17", "ru", "Когда распался СССР?", ["hist_ussr"]),
    ("q18", "ru", "В каком году закончилась Вторая мировая?", ["hist_wwii"]),
    ("q19", "ru", "Кем работает пользователь?", ["me_job"]),
    ("q20", "ru", "В каком городе живёт пользователь?", ["me_city"]),
    ("q21", "ru", "Чем увлекается пользователь?", ["me_hobby"]),
    ("q22", "ru", "Какой язык программирования любит пользователь?", ["me_lang", "tech_rust"]),
    ("q23", "en", "Who wrote the novel War and Peace?", ["auth_tolstoy"]),
    ("q24", "en", "What is the powerhouse of the cell?", ["sci_mito"]),
    ("q25", "en", "Which language has no garbage collector?", ["en_rust", "tech_rust"]),
    ("q26", "kz", "Қазақстанның астанасы қай қала?", ["cap_kz"]),
    ("q27", "kz", "Қазақтың ұлы ақыны кім?", ["kz_abai"]),
    ("q28", "ru", "Какой протокол работает поверх TCP?", ["tech_http"]),
]


def l2(v):
    v = mx.array(v)
    return (v / mx.maximum(mx.sqrt(mx.sum(v * v)), 1e-12)).tolist()


def embed(model, tok, text):
    inp = tok.batch_encode_plus([text], return_tensors="mlx", padding=True,
                                truncation=True, max_length=512)
    out = model(inp["input_ids"], attention_mask=inp["attention_mask"])
    emb = getattr(out, "text_embeds", None)
    if emb is None:
        lhs, mask = out.last_hidden_state, inp["attention_mask"][..., None]
        emb = (lhs * mask).sum(axis=1) / mx.maximum(mask.sum(axis=1), 1)
    return l2(emb[0].tolist())


def main():
    model, tok = load(MODEL)
    corpus = [{"id": i, "lang": l, "text": t, "emb": embed(model, tok, f"passage: {t}")}
              for (i, l, t) in CORPUS]
    queries = [{"id": i, "lang": l, "text": t, "relevant": rel,
                "emb": embed(model, tok, f"query: {t}")}
               for (i, l, t, rel) in QUERIES]
    out = {"model": MODEL, "dim": len(corpus[0]["emb"]), "corpus": corpus, "queries": queries}
    path = sys.argv[1] if len(sys.argv) > 1 else "bench_retrieval.json"
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"wrote {path}: dim={out['dim']}, corpus={len(corpus)}, queries={len(queries)}")


if __name__ == "__main__":
    main()

import os
import re
import json
from datetime import datetime
import neman_search
import build_synonyms
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = neman_search.SimpleSearchEngine()
engine.load_xml("neman.xml")

LAYOUT_EN = "qwertyuiop[]asdfghjkl;'zxcvbnm,./"
LAYOUT_RU = "йцукенгшщзхъфывапролджэячсмитьбю."
TRANS_EN_RU = str.maketrans(LAYOUT_EN, LAYOUT_RU)
TRANS_RU_EN = str.maketrans(LAYOUT_RU, LAYOUT_EN)
TRANSLIT_DICT = {
    "shh": "щ", "zh": "ж", "ch": "ч", "sh": "ш", "ts": "ц",
    "yu": "ю", "ya": "я", "yo": "ё", "a": "а", "b": "б",
    "v": "в", "g": "г", "d": "д", "e": "е", "z": "з",
    "i": "и", "y": "й", "k": "к", "l": "л", "m": "м",
    "n": "н", "o": "о", "p": "п", "r": "р", "s": "с",
    "t": "т", "u": "у", "f": "ф", "h": "х", "c": "ц",
    "x": "кс", "q": "к", "w": "в", "j": "дж",
}

COMPOUND_WORDS = {
    "аквамарис": "аква марис",
    "аквадетрим": "аква детрим",
    "аквалор": "аква лор",
}

_RU_SUFFIXES = tuple(sorted({
    "иями","ями","ами","ого","его","ому","ему","ыми","ими",
    "ах","ях","ов","ев","ая","яя","ое","ее","ые","ие",
    "ой","ей","ый","ий","ют","ят","ит","ет","ат",
    "ться","тся","ия","ии","ию",
    "а","я","о","е","ы","и","у","ю","й","ь",
}, key=len, reverse=True))

def _ru_stem(word: str) -> str:
    if len(word) <= 3:
        return word
    for suf in _RU_SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[:-len(suf)]
    return word

_PAIN_WORDS = frozenset({
    "боль","боли","болью","болей","болит","болят",
    "болело","болела","болели","больно","болезненно",
    "болезненный","болезненная","болезненное","болезненных",
    "болящий","болящая","болячка","болячки",
    "ноет","ноют","нытьё","нытье",
    "жжет","жжёт","жжение","жжения",
    "щиплет","щипит","щипание",
    "пульсирует","пульсация","ломит",
    "дергает","дёргает","стреляет","режет","колет",
    "першит","саднит",
})
_PAIN_STEMS = frozenset({"боляч","болезн"})

def _is_pain_word(w: str) -> bool:
    return w in _PAIN_WORDS or any(_ru_stem(w).startswith(s) for s in _PAIN_STEMS)

def _sub_matches(sub: dict, qw: list) -> bool:
    exact = sub.get("exact", frozenset())
    stems = sub.get("stems", ())
    for w in qw:
        if w in exact:
            return True
        ws = _ru_stem(w)
        if len(ws) >= 3 and any(ws.startswith(st) for st in stems):
            return True
    return False

def _rule_matches(rule: dict, qw: list) -> bool:
    if "all" in rule:
        return all(_sub_matches(s, qw) for s in rule["all"])
    return _sub_matches(rule, qw)

_SYMPTOM_RULES = [
    {"stems": ["голов"],
     "is_location": True,
     "groups": ["анальгетики","нпвс","жаропонижающие"]},

    {"exact": frozenset({"зуб","зуба","зубу","зубом","зубе",
                         "зубов","зубам","зубами","зубах"}),
     "stems": ["зубн"],
     "is_location": True,
     "groups": ["анальгетики","нпвс"]},

    {"stems": ["сустав","суставн"],
     "is_location": True,
     "groups": ["нпвс","нпвс_местно","хондропротекторы","анальгетики"]},

    {"stems": ["спин","поясниц","мышц"],
     "is_location": True,
     "groups": ["нпвс_местно","нпвс","спазмолитики"]},

    {"stems": ["сердц","сердечн"],
     "is_location": True,
     "groups": ["сердечные"]},

    {"exact": frozenset({"живот","живота","животу","животом","животе",
                         "желудок","желудком"}),
     "stems": ["желудочн","желудк"],
     "is_location": True,
     "groups": ["спазмолитики","ферменты","антациды"]},

    {"stems": ["горл"],
     "is_location": True,
     "groups": ["антисептики_горла"]},

    {"exact": frozenset({"ухо","уха","уху","ухом","ухе",
                         "уши","ушей","ушам","ушами","ушах"}),
     "stems": ["ушн"],
     "is_location": True,
     "groups": ["ушные"]},

    {"stems": ["глаз"],
     "is_location": True,
     "groups": ["глазные"]},

    {"stems": ["ног","рук","плеч","колен","локт","запяст"],
     "is_location": True,
     "groups": ["нпвс_местно","нпвс","анальгетики"]},

    {"stems": ["печен"],
     "is_location": True,
     "groups": ["гепатопротекторы"]},

    {"stems": ["мигрен"],
     "groups": ["анальгетики","нпвс","жаропонижающие"]},

    {"stems": ["варикоз"],
     "groups": ["венотоники"]},

    {"exact": frozenset({"геморрой","геморроя","геморрою","геморроем","геморрое"}),
     "stems": ["геморро"],
     "groups": ["от_геморроя"]},

    {"stems": ["аллерг"],
     "groups": ["антигистаминные"]},

    {"stems": ["давлен","гипертони","гипотони"],
     "groups": ["гипотензивные"]},

    {"stems": ["изжог"],
     "groups": ["антациды","ипп"]},

    {"stems": ["понос","диаре"],
     "groups": ["антидиарейные","сорбенты"]},

    {"stems": ["запор"],
     "groups": ["слабительные"]},

    {"stems": ["отравлен","тошнот","рвот"],
     "groups": ["сорбенты"]},

    {"stems": ["вздут","метеоризм"],
     "groups": ["ферменты","пробиотики"]},

    {"stems": ["дисбактериоз","дисбиоз"],
     "groups": ["пробиотики"]},

    {"stems": ["герпес"],
     "groups": ["противогерпетические"]},

    {"stems": ["простуд","грипп"],
     "exact": frozenset({"орви","орз"}),
     "groups": ["противовирусные","жаропонижающие"]},

    {"stems": ["иммун"],
     "groups": ["иммуномодуляторы","витамины"]},

    {"stems": ["температур","лихорадк"],
     "exact": frozenset({"жар","жара","жару","жаром"}),
     "groups": ["жаропонижающие"]},

    {"stems": ["стресс","тревог","невроз","паник"],
     "exact": frozenset({"нервы","нерв"}),
     "groups": ["седативные"]},

    {"stems": ["грибк","грибков"],
     "exact": frozenset({"грибок","грибка","грибку","грибком","грибке",
                         "грибковая","грибковое","грибковый"}),
     "groups": ["противогрибковые"]},

    {"stems": ["антибиотик","бактери"],
     "groups": ["антибиотики_пенициллины","антибиотики_макролиды",
                "антибиотики_цефалоспорины","антибиотики_фторхинолоны"]},

    {"all": [
        {"stems": ["кашл"], "exact": frozenset({"кашель"})},
        {"stems": ["сух"]}
     ],
     "groups": ["противокашлевые","муколитики"]},

    {"all": [
        {"stems": ["кашл"], "exact": frozenset({"кашель"})},
        {"stems": ["мокр","мокрот"]}
     ],
     "groups": ["муколитики","противокашлевые"]},

    {"stems": ["кашл"],
     "exact": frozenset({"кашель","кашля","кашлю","кашлем","кашле"}),
     "groups": ["муколитики","противокашлевые"]},

    {"exact": frozenset({"насморк","насморка","насморком",
                         "заложенность","заложен","заложена","сопли","сопля"}),
     "stems": ["насморк"],
     "groups": ["сосудосуживающие"]},

    {"stems": ["бессонниц"],
     "exact": frozenset({"спать","сплю","спала","спал","уснуть",
                         "засну","заснул","заснула","засыпать"}),
     "groups": ["снотворные","седативные"]},

    {"exact": frozenset({"рана","раны","ранку","ранки",
                         "порез","порезы","ожог","ожоги",
                         "царапина","царапины","ссадина","ссадины"}),
     "stems": ["ожог","порез","царапин","ссадин"],
     "groups": ["антисептики_ран"]},
]

_PAIN_FALLBACK_GROUPS = ["анальгетики","нпвс","жаропонижающие"]


def find_symptom_groups(raw_query: str) -> list:
    qw = raw_query.lower().strip().split()
    has_pain = any(_is_pain_word(w) for w in qw)
    matched: list = []
    location_hit = False

    def _add(groups):
        for g in groups:
            if g not in matched:
                matched.append(g)

    for rule in _SYMPTOM_RULES:
        if rule.get("is_location") and not has_pain:
            continue
        if _rule_matches(rule, qw):
            _add(rule["groups"])
            if rule.get("is_location"):
                location_hit = True

    if not location_hit and has_pain:
        _add(_PAIN_FALLBACK_GROUPS)

    return matched


JUNK_CATEGORY_NAMES = frozenset({
    "новая категория", "new category", "новые товары", "new products",
    "без категории", "uncategorized", "другое", "прочее", "разное",
    "все товары", "all products", "товары",
})

PHARMA_SYNONYMS = {}
if os.path.exists("synonyms.json"):
    try:
        with open("synonyms.json", "r", encoding="utf-8") as f:
            PHARMA_SYNONYMS = json.load(f)
    except Exception:
        pass

RECOMMENDATIONS = {}
DRUG_TO_GROUPS = {}
if os.path.exists("recommendations.json"):
    try:
        with open("recommendations.json", "r", encoding="utf-8") as f:
            RECOMMENDATIONS = json.load(f)
        for group_key, group_data in RECOMMENDATIONS.items():
            for drug in group_data.get("drugs", []):
                drug_low = drug.lower()
                if drug_low not in DRUG_TO_GROUPS:
                    DRUG_TO_GROUPS[drug_low] = []
                DRUG_TO_GROUPS[drug_low].append(group_key)
    except Exception:
        pass

DRUG_WORD_INDEX = {}
GROUP_WORD_INDEX = {}
GROUP_KEYS_WORDS = {}
REC_CACHE = {}

def _build_indices():
    global DRUG_WORD_INDEX, GROUP_WORD_INDEX, GROUP_KEYS_WORDS, REC_CACHE
    dw, gw, gkw, rc = {}, {}, {}, {}
    for drug_low in DRUG_TO_GROUPS:
        for word in drug_low.split():
            dw.setdefault(word, set()).add(drug_low)
    DRUG_WORD_INDEX = dw
    for group_key, group_data in RECOMMENDATIONS.items():
        clean_key = group_key.replace("_", " ").lower()
        label = group_data.get("label", "").lower()
        gkw[group_key] = frozenset(clean_key.split())
        for word in (clean_key + " " + label).split():
            gw.setdefault(word, set()).add(group_key)
    GROUP_WORD_INDEX = gw
    GROUP_KEYS_WORDS = gkw
    for drug_low in DRUG_TO_GROUPS:
        hits = engine.search(drug_low)
        if hits:
            rc[drug_low] = hits[:10]  
    REC_CACHE = rc

if RECOMMENDATIONS:
    _build_indices()


def normalize_query(text: str) -> str:
    t = text.lower().strip()
    for compound, split_form in COMPOUND_WORDS.items():
        t = t.replace(compound, split_form)
    return " ".join(t.split())


def name_relevance(query: str, name: str) -> float:
    q = query.lower().strip()
    n = name.lower().strip()
    if not q or not n:
        return 0.0
    if q == n:
        return 10.0
    if n.startswith(q + " ") or n == q:
        return 9.5
    q_words = q.split()
    n_words = n.split()
    n_words_set = set(n_words)
    exact_all = all(w in n_words_set for w in q_words)
    if exact_all:
        return 8.0 + 0.1 * len(q_words)
    prefix_all = all(any(nw.startswith(w) for nw in n_words) for w in q_words)
    if prefix_all:
        return 7.0 + 0.1 * len(q_words)
    exact_count = sum(1 for w in q_words if w in n_words_set)
    if exact_count > 0:
        return 4.0 + exact_count
    prefix_count = sum(1 for w in q_words if any(nw.startswith(w) for nw in n_words))
    if prefix_count > 0:
        return 2.0 + prefix_count
    return 0.0


def expand_query(query_text: str) -> str:
    query_lower = query_text.lower()
    query_words = set(query_lower.split())
    expanded = query_text
    for key, syns in PHARMA_SYNONYMS.items():
        key_words = key.split()
        if key_words and all(w in query_words for w in key_words):
            expanded += " " + " ".join(syns)
    words = []
    for w in expanded.split():
        if w not in words:
            words.append(w)
    return " ".join(words)


def switch_layout(text: str) -> str:
    return text.translate(TRANS_RU_EN) if re.search(r"[а-яё]", text) else text.translate(TRANS_EN_RU)


def translit_en_ru(text: str) -> str:
    res = text
    for eng, rus in TRANSLIT_DICT.items():
        res = res.replace(eng, rus)
    return res


def parse_price(price_str: str) -> float:
    try:
        clean = re.sub(r"[^\d.,]", "", price_str).replace(",", ".")
        return float(clean) if clean else 0.0
    except Exception:
        return 0.0


def format_item(hit: dict) -> dict:
    price_num = parse_price(hit["price"])
    oldprice_str = hit.get("oldprice", "").strip()
    oldprice_num = parse_price(oldprice_str) if oldprice_str else price_num
    if oldprice_num == 0.0:
        oldprice_num = price_num
    return {
        "id": hit["id"],
        "brand": None,
        "name": hit["original_name"],
        "url": hit.get("url", ""),
        "price": price_num,
        "oldprice": oldprice_num,
        "picture": hit.get("picture", ""),
        "currency": "сом",
        "is_presence": hit["in_stock"] == "true",
        "presence": None,
        "snippet": None,
        "label": None,
        "labels": [],
        "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000+00:00"),
        "params_data": {"has_discount": "true" if oldprice_num > price_num else "false"},
    }


def build_steps(normalized: str) -> list[str]:
    has_latin = bool(re.search(r"[a-z]", normalized))
    q_switched = switch_layout(normalized)
    q_translit = translit_en_ru(normalized) if has_latin else normalized
    steps = []
    if has_latin:
        if q_switched != normalized and re.search(r"[а-яё]", q_switched):
            steps.append(q_switched)
        if q_translit != normalized and re.search(r"[а-яё]", q_translit):
            steps.append(q_translit)
        steps.append(normalized)
    else:
        steps.append(normalized)
        if q_switched != normalized:
            steps.append(q_switched)
    return steps


def _symptom_hits(groups: list) -> list:
    seen_ids: set = set()
    hits: list = []
    for group_key in groups:
        group_data = RECOMMENDATIONS.get(group_key, {})
        for drug in group_data.get("drugs", []):
            for hit in REC_CACHE.get(drug.lower(), []):
                if hit["id"] not in seen_ids:
                    seen_ids.add(hit["id"])
                    hits.append(hit)
    return hits


def do_search(raw_query: str) -> tuple:
    normalized = normalize_query(raw_query)
    symptom_groups = find_symptom_groups(normalized)
    if symptom_groups:
        sym_hits = _symptom_hits(symptom_groups)
        engine_hits = engine.search(normalized)
        sym_ids = {h["id"] for h in sym_hits}
        extra = [h for h in engine_hits if h["id"] not in sym_ids]
        return sym_hits + extra, [], True  

    steps = build_steps(normalized)
    primary_hits = []
    used_step = normalized
    for q in steps:
        primary_hits = engine.search(q)
        if primary_hits:
            used_step = q
            break

    primary_hits.sort(
        key=lambda h: name_relevance(normalized, h["original_name"]),
        reverse=True,
    )

    secondary_hits = []
    for q in steps:
        expanded = expand_query(q)
        if expanded == q:
            continue
        expanded_hits = engine.search(expanded)
        if not expanded_hits:
            continue
        primary_ids = {h["id"] for h in primary_hits}
        raw_sec = [h for h in expanded_hits if h["id"] not in primary_ids]
        raw_sec.sort(
            key=lambda h: name_relevance(normalized, h["original_name"]),
            reverse=True,
        )
        secondary_hits = raw_sec
        break

    return primary_hits, secondary_hits, False 


def find_drug_groups(query_lower: str) -> set:
    found = set()
    query_words = set(query_lower.split())
    for word in query_words:
        found.update(GROUP_WORD_INDEX.get(word, set()))
    for group_key, ck_words in GROUP_KEYS_WORDS.items():
        if ck_words and ck_words.issubset(query_words):
            found.add(group_key)
    for word in query_words:
        if word in DRUG_TO_GROUPS:
            found.update(DRUG_TO_GROUPS[word])
    if query_lower in DRUG_TO_GROUPS:
        found.update(DRUG_TO_GROUPS[query_lower])
    for word in query_words:
        for drug_low in DRUG_WORD_INDEX.get(word, set()):
            found.update(DRUG_TO_GROUPS[drug_low])
    return found


def get_recommendations(search_query: str, main_hits: list, rec_limit: int = 5) -> tuple:
    if not RECOMMENDATIONS:
        return [], ""

    query_lower = normalize_query(search_query)
    symptom_grps = find_symptom_groups(query_lower)
    drug_grps = find_drug_groups(query_lower)
    found_groups: list = list(symptom_grps)
    for g in drug_grps:
        if g not in found_groups:
            found_groups.append(g)

    if not found_groups:
        return [], ""

    group_label = ""
    similar_drugs: list = []
    seen_drugs: set = set()
    for group_key in found_groups:
        group_data = RECOMMENDATIONS.get(group_key, {})
        if not group_label:
            group_label = group_data.get("label", "")
        for drug in group_data.get("drugs", []):
            drug_low = drug.lower()
            if drug_low not in query_lower and drug_low not in seen_drugs:
                similar_drugs.append(drug)
                seen_drugs.add(drug_low)

    main_ids = {h["id"] for h in main_hits}
    seen_rec_ids = set(main_ids)
    rec_items: list = []

    for drug in similar_drugs:
        if len(rec_items) >= rec_limit:
            break
        for hit in REC_CACHE.get(drug.lower(), []):
            if hit["id"] not in seen_rec_ids:
                seen_rec_ids.add(hit["id"])
                rec_items.append(format_item(hit))
                break

    return rec_items[:rec_limit], group_label


@app.get("/")
def search_api(
    request: Request,
    query: str = Query("", alias="query"),
    q: str = Query("", alias="q"),
    limit: int = Query(20),
    offset: int = Query(0),
    sort: str = Query("relevance"),
    t: list[str] = Query(None, alias="t[]"),
):
    raw_query = query.strip() or q.strip()
    if len(raw_query) < 2:
        return {
            "query": raw_query,
            "total": 0,
            "results": {"items": [], "categories": []},
            "recommendations": {"group": "", "items": []},
        }

    primary_hits, secondary_hits, is_symptom_search = do_search(raw_query)
    res_hits = primary_hits + secondary_hits

    seen_ids = set()
    deduped_hits = []
    for hit in res_hits:
        if hit["id"] not in seen_ids:
            seen_ids.add(hit["id"])
            deduped_hits.append(hit)
    res_hits = deduped_hits

    primary_cat_ids = set()
    for hit in primary_hits:
        for cid in hit.get("category_ids", "").split(","):
            cid = cid.strip()
            if cid:
                primary_cat_ids.add(cid)

    category_counts = {}
    for hit in res_hits:
        cids = hit.get("category_ids", "").split(",")
        cnames = hit.get("category_names", "").split("|")
        for cid, cname in zip(cids, cnames):
            cid, cname = cid.strip(), cname.strip()
            if not cid or not cname:
                continue
            if cname.lower().strip() in JUNK_CATEGORY_NAMES:
                continue
            if cid not in primary_cat_ids:
                continue
            if cid not in category_counts:
                category_counts[cid] = {"id": cid, "name": cname, "count": 0}
            category_counts[cid]["count"] += 1

    categories_list = sorted(category_counts.values(), key=lambda x: x["count"], reverse=True)

    if t:
        res_hits = [
            hit for hit in res_hits
            if any(
                cid.strip() in t
                for cid in hit.get("category_ids", "").split(",")
                if cid.strip()
            )
        ]

    if sort == "price.asc":
        res_hits.sort(key=lambda x: parse_price(x["price"]))
    elif sort == "price.desc":
        res_hits.sort(key=lambda x: parse_price(x["price"]), reverse=True)
    elif sort == "name.asc":
        res_hits.sort(key=lambda x: x["original_name"])
    elif sort == "name.desc":
        res_hits.sort(key=lambda x: x["original_name"], reverse=True)

    total_hits = len(res_hits)

    if is_symptom_search:
        rec_items, rec_group_label = [], ""
    else:
        rec_items, rec_group_label = get_recommendations(raw_query, res_hits, rec_limit=5)

    page_hits = res_hits[offset: offset + limit]
    items = [format_item(hit) for hit in page_hits]

    return {
        "query": raw_query,
        "total": total_hits,
        "results": {"items": items, "categories": categories_list},
        "recommendations": {"group": rec_group_label, "items": rec_items},
    }


@app.get("/api/reload")
def reload_catalog():
    try:
        build_synonyms.build_synonyms_file()
        global PHARMA_SYNONYMS, RECOMMENDATIONS, DRUG_TO_GROUPS
        if os.path.exists("synonyms.json"):
            with open("synonyms.json", "r", encoding="utf-8") as f:
                PHARMA_SYNONYMS = json.load(f)
        if os.path.exists("recommendations.json"):
            with open("recommendations.json", "r", encoding="utf-8") as f:
                RECOMMENDATIONS = json.load(f)
            DRUG_TO_GROUPS = {}
            for group_key, group_data in RECOMMENDATIONS.items():
                for drug in group_data.get("drugs", []):
                    drug_low = drug.lower()
                    if drug_low not in DRUG_TO_GROUPS:
                        DRUG_TO_GROUPS[drug_low] = []
                    DRUG_TO_GROUPS[drug_low].append(group_key)
            _build_indices()
    except Exception:
        pass
    total = engine.load_xml("neman.xml")
    return {"status": "success", "total_items": total}


@app.get("/demo", response_class=HTMLResponse)
def get_frontend():
    return """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Neman SearchCore</title>
        <style>
            *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
                background: #f0f2f5;
                min-height: 100vh;
                padding: 32px 16px;
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            .wrap {
                width: 100%;
                max-width: 720px;
            }
            .card {
                background: #fff;
                border-radius: 16px;
                box-shadow: 0 2px 12px rgba(0,0,0,0.08);
                padding: 28px 28px 20px;
                margin-bottom: 12px;
            }
            h1 {
                font-size: 22px;
                font-weight: 700;
                color: #1a1a2e;
                margin-bottom: 18px;
                text-align: center;
                letter-spacing: -0.3px;
            }
            .search-row {
                position: relative;
            }
            input[type="text"] {
                width: 100%;
                padding: 14px 48px 14px 18px;
                font-size: 16px;
                border: 2px solid #e8eaed;
                border-radius: 10px;
                outline: none;
                transition: border-color 0.15s;
                color: #1a1a2e;
            }
            input[type="text"]:focus { border-color: #4f8ef7; }
            .clear-btn {
                position: absolute;
                right: 14px;
                top: 50%;
                transform: translateY(-50%);
                background: none;
                border: none;
                cursor: pointer;
                color: #b0b4be;
                font-size: 18px;
                padding: 0;
                display: none;
                line-height: 1;
            }
            .clear-btn.visible { display: block; }
            .filters {
                display: flex;
                flex-wrap: wrap;
                gap: 7px;
                margin-top: 14px;
            }
            .filter-btn {
                padding: 5px 13px;
                border: 1.5px solid #d0d4de;
                border-radius: 20px;
                background: #fff;
                color: #555;
                cursor: pointer;
                font-size: 13px;
                transition: all 0.15s;
                white-space: nowrap;
            }
            .filter-btn:hover { border-color: #4f8ef7; color: #4f8ef7; }
            .filter-btn.active {
                background: #4f8ef7;
                border-color: #4f8ef7;
                color: #fff;
            }
            .results-area {
                margin-top: 8px;
            }
            .results-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 14px 0 8px;
                border-bottom: 1px solid #f0f2f5;
                margin-bottom: 4px;
            }
            .results-count {
                font-size: 13px;
                color: #8a90a0;
            }
            .sort-select {
                font-size: 13px;
                border: 1.5px solid #e8eaed;
                border-radius: 6px;
                padding: 4px 8px;
                color: #555;
                cursor: pointer;
                outline: none;
            }
            .product-item {
                display: flex;
                align-items: center;
                gap: 12px;
                padding: 11px 4px;
                border-bottom: 1px solid #f5f6f8;
                cursor: pointer;
                transition: background 0.1s;
                border-radius: 6px;
                text-decoration: none;
            }
            .product-item:hover { background: #f8f9fb; }
            .product-item.out-of-stock { opacity: 0.5; }
            .product-img {
                width: 44px;
                height: 44px;
                object-fit: contain;
                border-radius: 6px;
                background: #f5f6f8;
                flex-shrink: 0;
            }
            .product-img-placeholder {
                width: 44px;
                height: 44px;
                border-radius: 6px;
                background: #f0f2f5;
                flex-shrink: 0;
                display: flex;
                align-items: center;
                justify-content: center;
                color: #c8ccd8;
                font-size: 20px;
            }
            .product-info { flex: 1; min-width: 0; }
            .product-name {
                font-size: 14px;
                font-weight: 500;
                color: #1a1a2e;
                line-height: 1.4;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }
            .product-meta { font-size: 12px; color: #aab0be; margin-top: 2px; }
            .product-price-col { text-align: right; flex-shrink: 0; }
            .price-main { font-size: 15px; font-weight: 700; color: #2dba6e; }
            .price-old { font-size: 12px; color: #b0b4be; text-decoration: line-through; }
            .price-out { font-size: 13px; color: #f06060; font-weight: 500; }
            .badge-discount {
                display: inline-block;
                background: #fff0f0;
                color: #f06060;
                font-size: 11px;
                font-weight: 700;
                padding: 2px 6px;
                border-radius: 4px;
                margin-top: 2px;
            }
            .section-label {
                font-size: 11px;
                font-weight: 700;
                color: #aab0be;
                text-transform: uppercase;
                letter-spacing: 0.6px;
                margin: 16px 0 6px;
            }
            .rec-section {
                border-top: 2px dashed #eef0f4;
                padding-top: 12px;
                margin-top: 8px;
            }
            .empty-state {
                text-align: center;
                padding: 36px 16px;
                color: #aab0be;
                font-size: 15px;
            }
            .loader {
                text-align: center;
                padding: 16px;
                color: #aab0be;
                font-size: 13px;
            }
            .reload-btn {
                margin-top: 16px;
                width: 100%;
                background: #f0f2f5;
                color: #8a90a0;
                border: none;
                padding: 10px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 13px;
                transition: background 0.15s;
            }
            .reload-btn:hover { background: #e4e6eb; }
        </style>
    </head>
    <body>
        <div class="wrap">
            <div class="card">
                <h1>Search Engine</h1>
                <div class="search-row">
                    <input type="text" id="search-input" placeholder="Препарат, симптом, активное вещество..." autocomplete="off" spellcheck="false">
                    <button class="clear-btn" id="clear-btn" title="Очистить">✕</button>
                </div>
                <div class="filters" id="filters"></div>
            </div>

            <div id="results-area" style="display:none">
                <div class="results-header">
                    <span class="results-count" id="results-count"></span>
                    <select class="sort-select" id="sort-select">
                        <option value="relevance">По релевантности</option>
                        <option value="price.asc">Цена ↑</option>
                        <option value="price.desc">Цена ↓</option>
                        <option value="name.asc">А → Я</option>
                        <option value="name.desc">Я → А</option>
                    </select>
                </div>
                <div id="results-list"></div>
                <div id="scroll-anchor"></div>
                <div class="rec-section" id="rec-section" style="display:none">
                    <div class="section-label" id="rec-label">Похожие препараты</div>
                    <div id="rec-list"></div>
                </div>
                <div class="loader" id="loader" style="display:none">Загрузка...</div>
            </div>

            <button class="reload-btn" onclick="reloadXML()">Обновить базу товаров</button>
        </div>

        <script>
        const searchInput = document.getElementById('search-input');
        const clearBtn = document.getElementById('clear-btn');
        const filtersDiv = document.getElementById('filters');
        const resultsArea = document.getElementById('results-area');
        const resultsCount = document.getElementById('results-count');
        const resultsList = document.getElementById('results-list');
        const recSection = document.getElementById('rec-section');
        const recLabel = document.getElementById('rec-label');
        const recList = document.getElementById('rec-list');
        const sortSelect = document.getElementById('sort-select');
        const loader = document.getElementById('loader');
        const scrollAnchor = document.getElementById('scroll-anchor');

        const FETCH_LIMIT = 20;
        let state = {
            query: '',
            filter: null,
            sort: 'relevance',
            offset: 0,
            hasMore: false,
            loading: false,
            version: 0,
        };

        function renderItem(item) {
            const a = document.createElement('a');
            a.className = 'product-item' + (!item.is_presence ? ' out-of-stock' : '');
            a.href = item.url || '#';
            if (item.url) a.target = '_blank';
            a.rel = 'noopener';

            const hasDiscount = item.params_data?.has_discount === 'true' && item.oldprice > item.price;
            const discountPct = hasDiscount ? Math.round((1 - item.price / item.oldprice) * 100) : 0;

            const imgEl = item.picture
                ? `<img class="product-img" src="${item.picture}" alt="" loading="lazy" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">`
                  + `<div class="product-img-placeholder" style="display:none">img</div>`
                : `<div class="product-img-placeholder">img</div>`;

            const priceEl = item.is_presence
                ? `<div class="price-main">${item.price.toLocaleString('ru')} сом</div>`
                  + (hasDiscount ? `<div class="price-old">${item.oldprice.toLocaleString('ru')}</div><div class="badge-discount">−${discountPct}%</div>` : '')
                : `<div class="price-out">Нет в наличии</div>`;

            a.innerHTML = imgEl + `
                <div class="product-info">
                    <div class="product-name">${item.name}</div>
                </div>
                <div class="product-price-col">${priceEl}</div>`;
            return a;
        }

        async function apiFetch(query, filter, sort, offset) {
            let url = `/?query=${encodeURIComponent(query)}&limit=${FETCH_LIMIT}&offset=${offset}&sort=${sort}`;
            if (filter) url += `&t[]=${encodeURIComponent(filter)}`;
            const r = await fetch(url);
            return r.json();
        }

        async function doSearch(keepFilters = false) {
            const v = ++state.version;
            state.offset = 0;
            state.hasMore = false;
            resultsList.innerHTML = '';
            recSection.style.display = 'none';

            if (!state.query || state.query.length < 2) {
                resultsArea.style.display = 'none';
                if (!keepFilters) { filtersDiv.innerHTML = ''; state.filter = null; }
                return;
            }

            loader.style.display = 'block';
            resultsArea.style.display = 'block';

            try {
                const data = await apiFetch(state.query, state.filter, state.sort, 0);
                if (v !== state.version) return;
                loader.style.display = 'none';

                if (!keepFilters) {
                    renderFilters(data.results.categories || []);
                }

                resultsCount.textContent = `Найдено: ${data.total}`;
                state.hasMore = data.results.items.length === FETCH_LIMIT;
                state.offset = data.results.items.length;

                if (data.results.items.length === 0) {
                    resultsList.innerHTML = '<div class="empty-state">Ничего не найдено</div>';
                } else {
                    data.results.items.forEach(item => resultsList.appendChild(renderItem(item)));
                }

                const recs = data.recommendations;
                if (recs && recs.items && recs.items.length > 0) {
                    recLabel.textContent = recs.group ? `Похожие: ${recs.group}` : 'Похожие препараты';
                    recList.innerHTML = '';
                    recs.items.forEach(item => recList.appendChild(renderItem(item)));
                    recSection.style.display = 'block';
                }
            } catch (e) {
                if (v === state.version) loader.style.display = 'none';
            }
        }

        async function loadMore() {
            if (!state.hasMore || state.loading || !state.query || state.query.length < 2) return;
            state.loading = true;
            const v = state.version;
            loader.style.display = 'block';
            try {
                const data = await apiFetch(state.query, state.filter, state.sort, state.offset);
                if (v !== state.version) { state.loading = false; return; }
                loader.style.display = 'none';
                if (data.results.items.length < FETCH_LIMIT) state.hasMore = false;
                data.results.items.forEach(item => resultsList.appendChild(renderItem(item)));
                state.offset += data.results.items.length;
            } catch (e) {
                if (v === state.version) loader.style.display = 'none';
            }
            state.loading = false;
        }

        function renderFilters(categories) {
            filtersDiv.innerHTML = '';
            state.filter = null;
            if (!categories || categories.length <= 1) return;
            categories.forEach(cat => {
                const btn = document.createElement('button');
                btn.className = 'filter-btn';
                btn.dataset.catId = cat.id;
                btn.textContent = `${cat.name} (${cat.count})`;
                btn.onclick = async () => {
                    const wasActive = state.filter === cat.id;
                    state.filter = wasActive ? null : cat.id;
                    document.querySelectorAll('.filter-btn').forEach(b => {
                        b.classList.toggle('active', b.dataset.catId === state.filter);
                    });
                    await doSearch(true);
                };
                filtersDiv.appendChild(btn);
            });
        }

        let debounceTimer = null;
        searchInput.addEventListener('input', e => {
            const val = e.target.value.trim();
            state.query = val;
            clearBtn.classList.toggle('visible', val.length > 0);
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(() => doSearch(false), 150);
        });

        clearBtn.addEventListener('click', () => {
            searchInput.value = '';
            state.query = '';
            state.filter = null;
            clearBtn.classList.remove('visible');
            filtersDiv.innerHTML = '';
            resultsArea.style.display = 'none';
            resultsList.innerHTML = '';
            searchInput.focus();
        });

        sortSelect.addEventListener('change', () => {
            state.sort = sortSelect.value;
            doSearch(true);
        });

        const observer = new IntersectionObserver(entries => {
            if (entries[0].isIntersecting) loadMore();
        }, { rootMargin: '150px' });
        observer.observe(scrollAnchor);

        async function reloadXML() {
            try {
                const res = await fetch('/api/reload');
                const data = await res.json();
                alert(`База обновлена. Товаров: ${data.total_items}`);
            } catch (e) {}
        }
        </script>
    </body>
    </html>
    """

"""S 도메인(가상 스포츠 기록실 2026) 데이터 패키지 생성기. 시드 고정이라 다시 실행해도 같은 결과가 나옴

  python data/S/build/make_sports_pkg.py

만드는 것(data/S/): schema.json, scoring.yaml, test_seed.json, test_dataset.json(처음에는 시드와 같음),
team_item_template.json, corpus/*.md, corpus_manifest.json, corpus_updates/(6회차 시연용 결승 결과),
train_seed.jsonl, tools_spec.md, build/sports_db.json
강사 전용(solutions/data/S/): test_pool.json(현장 보정·교체 풀)

리그·구단·선수·경기는 모두 가상임. 실존 구단·선수·대회와 관계없음
"""
from __future__ import annotations

import datetime as dt
import itertools
import json
import random
from pathlib import Path

import s_rules as R

HERE = Path(__file__).resolve().parent
DATA = HERE.parent
SOL = HERE.parents[2] / "solutions" / "data" / "S"
SNAPSHOT = "2026-11-08"            # 기록실 기준일. 이 날 이후 결과는 코퍼스에 없음
LICENSE = "CC BY 4.0"
SOURCE = "강사 제작 가상 스포츠 기록 DB"
LG_B, LG_E, LG_W = "한빛프로야구", "넥서스리그", "월드인비테이셔널"
SEED = 7

# ------------------------------------------------------------------ 구단(가상)
# id, 정식 이름, 약칭, 연고·소속, 전력, 쓰임(eval 평가 문항 / train 학습 문항)
B_TEAMS = [
    ("b01", "청주 비룡", "비룡", "청주", 0.60, "eval"),
    ("b02", "포항 해일", "해일", "포항", 0.55, "eval"),
    ("b03", "전주 백호", "백호", "전주", 0.52, "eval"),
    ("b04", "목포 고래", "고래", "목포", 0.50, "eval"),
    ("b05", "안동 두루미", "두루미", "안동", 0.49, "train"),
    ("b06", "춘천 반달", "반달", "춘천", 0.46, "train"),
    ("b07", "제주 돌풍", "돌풍", "제주", 0.45, "train"),
    ("b08", "원주 산매", "산매", "원주", 0.43, "train"),
]
E_TEAMS = [
    ("e01", "노바 크라운", "노바", "넥서스리그", 0.62, "eval"),
    ("e02", "비룡 게이밍", "비룡", "넥서스리그", 0.57, "eval"),      # 프로야구 '청주 비룡'과 같은 모기업. 약칭이 겹침
    ("e03", "블루 하운즈", "하운즈", "넥서스리그", 0.54, "eval"),
    ("e04", "레드 펄스", "펄스", "넥서스리그", 0.51, "eval"),
    ("e05", "아이언 팽", "팽", "넥서스리그", 0.49, "eval"),
    ("e06", "스텔라 나인", "스텔라", "넥서스리그", 0.50, "train"),
    ("e07", "볼트 원", "볼트", "넥서스리그", 0.47, "train"),
    ("e08", "오로라 게이밍", "오로라", "넥서스리그", 0.45, "train"),
    ("e09", "타이드 브레이커", "타이드", "넥서스리그", 0.44, "train"),
    ("e10", "실버 아울스", "아울스", "넥서스리그", 0.41, "train"),
]
W_FOREIGN = [  # 해외 가상 리그 팀(월드 인비테이셔널 참가)
    ("w01", "폴라리스 유니언", "북부연합", 0.60), ("w02", "크림슨 하이브", "북부연합", 0.52), ("w03", "프로스트 바인", "북부연합", 0.45),
    ("w04", "베르데 레기온", "서부리그", 0.58), ("w05", "솔라 앵커", "서부리그", 0.50), ("w06", "그레이 쿼츠", "서부리그", 0.43),
    ("w07", "진저 드래곤스", "동부리그", 0.59), ("w08", "옥스 블레이드", "동부리그", 0.51), ("w09", "모노 래빗", "동부리그", 0.42),
    ("w10", "코랄 시그널", "남부연합", 0.48), ("w11", "선셋 바이퍼", "남부연합", 0.44), ("w12", "마블 웨이브", "남부연합", 0.40),
]
SURN = list("김이박최정강조윤장임한오서신권황안송유홍전고문양손배백허남심노하곽성차주우구민류나진지엄채원천방공현함변염여추도소석선설마길연위표명기반왕금옥육인맹제모탁국어은편용")
GIVEN = ["도윤", "서준", "하준", "은호", "시우", "지호", "유찬", "건우", "우진", "선율", "태오", "이안", "도겸", "재윤", "승현", "민혁",
         "다온", "로운", "한결", "시헌", "준서", "윤재", "태린", "가람", "범수", "지환", "규민", "산하", "우람", "해솔", "보겸", "찬영"]
HANDLES = ["Quill", "Brine", "Tallow", "Juno", "Hollow", "Dalbit", "Narae", "Gyeol", "Sori", "Vesper", "Kiln", "Mosk", "Ripple",
           "Fennel", "Corvo", "Lattice", "Sable", "Pebble", "Onyx", "Wren", "Haze", "Tundra", "Yarrow", "Glint", "Morrow", "Bramble",
           "Cinder", "Ochre", "Saffron", "Thistle", "Umber", "Vale", "Whit", "Zinnia", "Arbor", "Basalt", "Cobble", "Drift", "Ember",
           "Flint", "Gale", "Heron", "Ivo", "Jasper", "Knoll", "Lumen", "Mire", "Nimbus", "Orrin", "Pike"]
E_POS = ["선봉", "유격", "중앙", "사수", "지원"]


def d(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def fmt_md(x: dt.date) -> str:
    return f"{x.month}월 {x.day}일"


def _batchim(w: str) -> bool:
    c = w.strip()[-1]
    if "가" <= c <= "힣":
        return (ord(c) - 0xAC00) % 28 != 0
    if c.isdigit():
        return c in "013678"
    return c.lower() in "lmn"


def iga(w: str) -> str:
    return w + ("이" if _batchim(w) else "가")


def eun(w: str) -> str:
    return w + ("은" if _batchim(w) else "는")


def eul(w: str) -> str:
    return w + ("을" if _batchim(w) else "를")


def ro(w: str) -> str:
    c = w.strip()[-1]
    rieul = ("가" <= c <= "힣" and (ord(c) - 0xAC00) % 28 == 8) or c in "178" or c.lower() == "l"
    return w + ("으로" if _batchim(w) and not rieul else "로")


def wa(w: str) -> str:
    return w + ("과" if _batchim(w) else "와")


# ------------------------------------------------------------------ 한빛프로야구 시즌 모의
def circle_rounds(ids: list[str]) -> list[list[tuple[str, str]]]:
    n, rot = len(ids), ids[:]
    rounds = []
    for _ in range(n - 1):
        rounds.append([(rot[i], rot[n - 1 - i]) for i in range(n // 2)])
        rot = [rot[0]] + [rot[-1]] + rot[1:-1]
    return rounds


def play_ball(rng: random.Random, home: str, away: str, st: dict) -> tuple[int, int]:
    p = st[home] / (st[home] + st[away]) + 0.03
    if rng.random() < 0.02:
        r = rng.randint(1, 6)
        return r, r
    w, l = rng.randint(3, 11), None
    l = rng.randint(0, w - 1)
    return (w, l) if rng.random() < p else (l, w)


def sim_baseball(rng: random.Random) -> dict:
    ids = [t[0] for t in B_TEAMS]
    st = {t[0]: t[4] for t in B_TEAMS}
    base = circle_rounds(ids)
    rounds = []
    for k in range(12):
        for rd in base:
            rounds.append([(a, b) if k % 2 == 0 else (b, a) for a, b in rd])
    days = [d("2026-03-28") + dt.timedelta(i) for i in range((d("2026-09-27") - d("2026-03-28")).days + 1)]
    days = [x for x in days if x.weekday() != 0]
    step = (len(days) - 1) / (len(rounds) - 1)
    games = []
    for i, rd in enumerate(rounds):
        day = days[round(i * step)]
        for h, a in rd:
            hs, as_ = play_ball(rng, h, a, st)
            games.append({"date": day.isoformat(), "home": h, "away": a, "hs": hs, "as": as_})
    return {"games": games}


def standings(games: list[dict], until: str | None = None) -> list[dict]:
    rec = {t[0]: {"id": t[0], "w": 0, "l": 0, "d": 0, "rs": 0, "ra": 0} for t in B_TEAMS}
    h2h: dict[tuple[str, str], int] = {}
    for g in games:
        if until and g["date"] > until:
            continue
        for me, op, ms, os_ in ((g["home"], g["away"], g["hs"], g["as"]), (g["away"], g["home"], g["as"], g["hs"])):
            r = rec[me]
            r["rs"] += ms
            r["ra"] += os_
            if ms > os_:
                r["w"] += 1
                h2h[(me, op)] = h2h.get((me, op), 0) + 1
            elif ms < os_:
                r["l"] += 1
            else:
                r["d"] += 1
    rows = list(rec.values())
    for r in rows:
        r["pct"] = r["w"] / (r["w"] + r["l"]) if r["w"] + r["l"] else 0
    rows.sort(key=lambda r: (-round(r["pct"], 6), -(r["rs"] - r["ra"])))
    # 승률이 같으면 상대 전적으로 다시 정렬
    for i in range(len(rows) - 1):
        a, b = rows[i], rows[i + 1]
        if round(a["pct"], 6) == round(b["pct"], 6) and h2h.get((b["id"], a["id"]), 0) > h2h.get((a["id"], b["id"]), 0):
            rows[i], rows[i + 1] = b, a
    lead = rows[0]
    for k, r in enumerate(rows, 1):
        r["rank"] = k
        r["pct_s"] = R.winrate(r["w"], r["l"])
        r["gb"] = "-" if k == 1 else R.games_behind((lead["w"], lead["l"]), (r["w"], r["l"]))
    return rows


def sim_series(rng: random.Random, hi: str, lo: str, need: int, start: dt.date, st: dict) -> dict:
    gs, wins = [], {hi: 0, lo: 0}
    day = start
    k = 0
    while max(wins.values()) < need:
        home = hi if k % 4 in (0, 1) else lo
        away = lo if home == hi else hi
        hs, as_ = play_ball(rng, home, away, st)
        while hs == as_:
            hs, as_ = play_ball(rng, home, away, st)
        win = home if hs > as_ else away
        wins[win] += 1
        gs.append({"no": k + 1, "date": day.isoformat(), "home": home, "away": away, "hs": hs, "as": as_, "win": win})
        k += 1
        day += dt.timedelta(1 if k % 2 else 2)
    winner = max(wins, key=wins.get)
    loser = lo if winner == hi else hi
    return {"hi": hi, "lo": lo, "games": gs, "winner": winner, "loser": loser,
            "score": f"{wins[winner]}:{wins[loser]}", "end": gs[-1]["date"]}


def sim_post(rng: random.Random, table: list[dict]) -> dict:
    st = {t[0]: t[4] for t in B_TEAMS}
    r = [x["id"] for x in table]
    wc = sim_series(rng, r[2], r[3], 2, d("2026-10-03"), st)
    po = sim_series(rng, r[1], wc["winner"], 3, d(wc["end"]) + dt.timedelta(2), st)
    ks = sim_series(rng, r[0], po["winner"], 4, d(po["end"]) + dt.timedelta(3), st)
    return {"준플레이오프": wc, "플레이오프": po, "한빛시리즈": ks}


def make_b_players(rng: random.Random) -> list[dict]:
    used, players = set(), []
    pid = 0
    for t in B_TEAMS:
        for role in ("타자", "타자", "투수"):
            while True:
                nm = rng.choice(SURN) + rng.choice(GIVEN)
                if nm not in used:
                    used.add(nm)
                    break
            pid += 1
            p = {"id": f"bp{pid:02d}", "name": nm, "team": t[0], "role": role, "use": t[5]}
            if role == "타자":
                ab = rng.randint(420, 560)
                avg = rng.uniform(0.255, 0.345)
                p.update(g=rng.randint(118, 144) // 1, ab=ab, h=round(ab * avg), hr=rng.randint(6, 41))
                p["rbi"] = min(130, p["hr"] * 2 + rng.randint(28, 62))
                p["pos"] = rng.choice(["1루수", "2루수", "유격수", "3루수", "좌익수", "중견수", "우익수", "지명타자", "포수"])
                p["g"] = min(p["g"], 84)
                p["avg"] = R.batting_avg(p["h"], p["ab"])
            else:
                outs = rng.randint(330, 520)
                ip = f"{outs // 3}.{outs % 3}"
                era_t = rng.uniform(2.15, 4.85)
                er = round(era_t * outs / 27)
                p.update(pos="선발투수", g=rng.randint(22, 29), w=rng.randint(6, 17), l=rng.randint(4, 12), ip=ip, er=er,
                         so=rng.randint(88, 205))
                p["era"] = R.era(er, ip)
            players.append(p)
    return players


# ------------------------------------------------------------------ 넥서스리그·월드 인비테이셔널 모의
def bo(rng: random.Random, a: str, b: str, need: int, st: dict) -> tuple[str, str, str]:
    wa = wb = 0
    while wa < need and wb < need:
        if rng.random() < st[a] / (st[a] + st[b]):
            wa += 1
        else:
            wb += 1
    return (a, b, f"{wa}:{wb}") if wa > wb else (b, a, f"{wb}:{wa}")


def sim_split(rng: random.Random, start: str, final_day: str) -> dict:
    ids = [t[0] for t in E_TEAMS]
    st = {t[0]: t[4] + rng.uniform(-0.03, 0.03) for t in E_TEAMS}
    rec = {i: {"id": i, "mw": 0, "ml": 0, "sw": 0, "sl": 0} for i in ids}
    matches = []
    day = d(start)
    for rd in circle_rounds(ids) * 2:
        for a, b in rd:
            w, l, sc = bo(rng, a, b, 2, st)
            sw, sl = (int(x) for x in sc.split(":"))
            rec[w]["mw"] += 1; rec[l]["ml"] += 1
            rec[w]["sw"] += sw; rec[w]["sl"] += sl; rec[l]["sw"] += sl; rec[l]["sl"] += sw
            matches.append({"date": day.isoformat(), "win": w, "lose": l, "score": sc})
        day += dt.timedelta(4 if day.weekday() in (2, 3) else 3)
    rows = sorted(rec.values(), key=lambda r: (-r["mw"], -(r["sw"] - r["sl"]), r["id"]))
    for k, r in enumerate(rows, 1):
        r["rank"] = k
        r["sd"] = r["sw"] - r["sl"]
    top = [r["id"] for r in rows[:4]]
    fd = d(final_day)
    sf1 = bo(rng, top[0], top[3], 3, st)
    sf2 = bo(rng, top[1], top[2], 3, st)
    fin = bo(rng, sf1[0], sf2[0], 3, st)
    po = {"4강 1": {"date": (fd - dt.timedelta(8)).isoformat(), "win": sf1[0], "lose": sf1[1], "score": sf1[2]},
          "4강 2": {"date": (fd - dt.timedelta(7)).isoformat(), "win": sf2[0], "lose": sf2[1], "score": sf2[2]},
          "결승": {"date": fd.isoformat(), "win": fin[0], "lose": fin[1], "score": fin[2]}}
    semis_lost = sorted([sf1[1], sf2[1]], key=lambda i: rec[i]["rank"])
    final_rank = [fin[0], fin[1]] + semis_lost
    return {"table": rows, "matches": matches, "playoffs": po, "final_rank": final_rank}


def make_e_players(rng: random.Random) -> list[dict]:
    pool = HANDLES[:]
    rng.shuffle(pool)
    out, k = [], 0
    for t in E_TEAMS:
        ace = rng.randrange(5)
        for j, pos in enumerate(E_POS):
            k += 1
            kills = rng.randint(30, 120) if pos in ("중앙", "사수") else rng.randint(12, 60)
            deaths = rng.randint(18, 55)
            assists = rng.randint(60, 190)
            out.append({"id": f"ep{k:02d}", "name": pool.pop(), "team": t[0], "pos": pos, "k": kills, "d": deaths, "a": assists,
                        "kda": R.kda(kills, deaths, assists), "pog": rng.randint(100, 900) // 100 * 100 if j == ace else rng.randint(0, 4) * 100,
                        "ace": j == ace, "use": t[5]})
    return out


def sim_wiv(rng: random.Random, nxl_top4: list[str]) -> dict:
    st = {t[0]: t[4] for t in E_TEAMS} | {w[0]: w[3] for w in W_FOREIGN}
    teams = nxl_top4 + [w[0] for w in W_FOREIGN]
    pots = [teams[0::4], teams[1::4], teams[2::4], teams[3::4]]
    groups = {g: [] for g in "ABCD"}
    for pot in pots:
        p = pot[:]
        rng.shuffle(p)
        for g, t in zip("ABCD", p):
            groups[g].append(t)
    gtab, gday = {}, d("2026-10-15")
    for g, mem in groups.items():
        rec = {t: {"id": t, "w": 0, "l": 0} for t in mem}
        for a, b in itertools.combinations(mem, 2):
            for _ in range(2):
                w, l, _sc = bo(rng, a, b, 1, st)
                rec[w]["w"] += 1; rec[l]["l"] += 1
        rows = sorted(rec.values(), key=lambda r: (-r["w"], r["id"]))
        for k, r in enumerate(rows, 1):
            r["rank"] = k
        gtab[g] = rows
    adv = {g: [r["id"] for r in rows[:2]] for g, rows in gtab.items()}
    qf_pairs = [(adv["A"][0], adv["B"][1]), (adv["B"][0], adv["A"][1]), (adv["C"][0], adv["D"][1]), (adv["D"][0], adv["C"][1])]
    qdays = ["2026-10-31", "2026-10-31", "2026-11-01", "2026-11-01"]
    qf = [dict(zip(("win", "lose", "score"), bo(rng, a, b, 3, st)), date=qd) for (a, b), qd in zip(qf_pairs, qdays)]
    sf = [dict(zip(("win", "lose", "score"), bo(rng, qf[0]["win"], qf[1]["win"], 3, st)), date="2026-11-07"),
          dict(zip(("win", "lose", "score"), bo(rng, qf[2]["win"], qf[3]["win"], 3, st)), date="2026-11-08")]
    fin = dict(zip(("win", "lose", "score"), bo(rng, sf[0]["win"], sf[1]["win"], 3, st)), date="2026-11-14")
    return {"groups": gtab, "qf": qf, "sf": sf, "final": fin}


# ------------------------------------------------------------------ 이름 찾기
def build_db() -> dict:
    for seed in range(SEED, SEED + 400):
        rng = random.Random(seed)
        bb = sim_baseball(rng)
        table = standings(bb["games"])
        pcts = [r["pct_s"] for r in table]
        if len(set(pcts)) < len(pcts):
            continue
        post = sim_post(rng, table)
        bpl = make_b_players(rng)
        spring = sim_split(rng, "2026-01-14", "2026-04-19")
        summer = sim_split(rng, "2026-06-10", "2026-08-30")
        epl = make_e_players(rng)
        wiv = sim_wiv(rng, summer["final_rank"])
        nx = {t[0] for t in E_TEAMS}
        hrs = sorted((p["hr"] for p in bpl if p["role"] == "타자"), reverse=True)
        if hrs[0] == hrs[1]:
            continue
        # 6회차 시연이 살도록: 결승에 넥서스리그 팀이 올라가고, 4강까지 결과가 기준일 안에 있어야 함
        if not ({wiv["sf"][0]["win"], wiv["sf"][1]["win"]} & nx):
            continue
        if spring["final_rank"][0] == summer["final_rank"][0]:
            continue
        return {"seed": seed, "baseball": {"games": bb["games"], "table": table, "post": post, "players": bpl,
                                           "monthly": {m: standings(bb["games"], m) for m in
                                                       ("2026-04-30", "2026-05-31", "2026-06-30", "2026-07-31", "2026-08-31")}},
                "esports": {"spring": spring, "summer": summer, "players": epl}, "wiv": wiv}
    raise RuntimeError("조건을 만족하는 시드가 없음")


TEAM = {t[0]: {"id": t[0], "name": t[1], "short": t[2], "home": t[3], "use": t[5], "league": LG_B} for t in B_TEAMS}
TEAM |= {t[0]: {"id": t[0], "name": t[1], "short": t[2], "home": t[3], "use": t[5], "league": LG_E} for t in E_TEAMS}
TEAM |= {w[0]: {"id": w[0], "name": w[1], "short": w[1].split()[0], "home": w[2], "use": "none", "league": LG_W} for w in W_FOREIGN}


def nm(i: str) -> str:
    return TEAM[i]["name"]


# ------------------------------------------------------------------ 코퍼스 문서
def doc(doc_id: str, title: str, body: str, as_of: str) -> dict:
    return {"doc_id": doc_id, "title": title, "as_of": as_of, "md": f"# {title}\n\n> 기록 기준일 {as_of} · 가상 리그 기록(실존 구단·선수와 무관)\n\n{body.strip()}\n"}


def baseball_docs(db: dict) -> list[dict]:
    b = db["baseball"]
    tab, post, pl = b["table"], b["post"], b["players"]
    out = []
    out.append(doc("league_hpb", "한빛프로야구 리그 안내", f"""
한빛프로야구는 8개 구단이 참가하는 가상 프로야구 리그이다. 2026 시즌은 3월 28일에 개막해 9월 27일에 정규시즌을 마쳤다.

## 경기 방식
- 구단마다 다른 7개 구단과 12번씩 맞붙어 정규시즌 84경기를 치름. 월요일에는 경기를 하지 않음
- 연장은 12회까지이며 12회가 끝나도 같으면 무승부로 기록함

## 순위 결정
- 순위는 승률로 정함. 승률 = 승 ÷ (승 + 패)이며 무승부는 계산에서 뺌
- 승률이 같으면 두 팀의 상대 전적이 앞선 팀이 위. 상대 전적도 같으면 득실차로 정함
- 게임차는 1위 팀과의 차이 = ((1위 승 - 해당 팀 승) + (해당 팀 패 - 1위 패)) ÷ 2

## 포스트시즌
- 준플레이오프 : 3위와 4위, 3전 2선승
- 플레이오프 : 2위와 준플레이오프 승자, 5전 3선승
- 한빛시리즈 : 1위와 플레이오프 승자, 7전 4선승. 이긴 팀이 그해 우승팀임
""", "2026-09-27"))

    rows = "\n".join(f"| {r['rank']} | {nm(r['id'])} | {r['w'] + r['l'] + r['d']} | {r['w']} | {r['l']} | {r['d']} | {r['pct_s']} | {r['gb']} | {r['rs']} | {r['ra']} |" for r in tab)
    monthly = []
    for m, t in b["monthly"].items():
        mm = d(m)
        order = ", ".join(f"{x['rank']}위 {nm(x['id'])}({x['w']}승 {x['l']}패)" for x in t)
        monthly.append(f"- **{mm.month}월 말({m})** : {order}")
    hr = sorted([p for p in pl if p["role"] == "타자"], key=lambda p: -p["hr"])[0]
    era_k = sorted([p for p in pl if p["role"] == "투수"], key=lambda p: float(p["era"]))[0]
    out.append(doc("season_hpb_2026", "2026 한빛프로야구 정규시즌", f"""
2026 한빛프로야구 정규시즌은 {iga(nm(tab[0]['id']))} {tab[0]['w']}승 {tab[0]['l']}패 {tab[0]['d']}무, 승률 {tab[0]['pct_s']}로 1위를 차지했다.
2위 {nm(tab[1]['id'])}에는 {tab[1]['gb']}게임 앞섰다. 포스트시즌에는 1~4위인 {', '.join(nm(r['id']) for r in tab[:4])} 네 구단이 나갔다.

## 최종 순위(9월 27일)

| 순위 | 구단 | 경기 | 승 | 패 | 무 | 승률 | 게임차 | 득점 | 실점 |
|---|---|---|---|---|---|---|---|---|---|
{rows}

## 월별 순위 변화
월말 기준 순위임. 시즌 중 순위는 그날까지의 경기만으로 계산함

{chr(10).join(monthly)}

## 시즌 주요 기록
- 홈런 1위 : {hr['name']}({nm(hr['team'])}) {hr['hr']}개
- 평균자책점 1위 : {era_k['name']}({nm(era_k['team'])}) {era_k['era']}
- 개인 기록 전체는 '2026 한빛프로야구 개인 기록 순위' 문서를 봄
""", "2026-09-27"))

    def series_md(nm_s: str, s: dict) -> str:
        lines = [f"| {g['no']}차전 | {fmt_md(d(g['date']))} | {nm(g['home'])} {g['hs']} : {g['as']} {nm(g['away'])} | {nm(g['win'])} |" for g in s["games"]]
        return (f"## {nm_s}\n{nm(s['hi'])} 대 {nm(s['lo'])}. {iga(nm(s['winner']))} 시리즈 {ro(s['score'])} 이김({fmt_md(d(s['end']))} 종료)\n\n"
                "| 경기 | 날짜 | 결과(홈 먼저) | 승리 |\n|---|---|---|---|\n" + "\n".join(lines))
    champ = post["한빛시리즈"]
    out.append(doc("post_hpb_2026", "2026 한빛프로야구 포스트시즌", f"""
2026 포스트시즌은 10월 3일 준플레이오프로 시작해 {fmt_md(d(champ['end']))} 한빛시리즈로 끝났다.
**2026 시즌 우승팀은 {nm(champ['winner'])}**이며, 한빛시리즈에서 {eul(nm(champ['loser']))} {ro(champ['score'])} 꺾었다.

{series_md('준플레이오프(3전 2선승)', post['준플레이오프'])}

{series_md('플레이오프(5전 3선승)', post['플레이오프'])}

{series_md('한빛시리즈(7전 4선승)', champ)}
""", champ["end"]))

    def top(key, rev=True, role="타자", n=5, fmt=str):
        ps = sorted([p for p in pl if p["role"] == role], key=lambda p: (float(p[key]) if isinstance(p[key], str) else p[key]), reverse=rev)[:n]
        return "\n".join(f"| {k} | {p['name']} | {nm(p['team'])} | {fmt(p[key])} |" for k, p in enumerate(ps, 1))
    out.append(doc("leaders_hpb_2026", "2026 한빛프로야구 개인 기록 순위", f"""
정규시즌 최종 기록이다. 각 부문 상위 5명만 적음. 타율은 규정 타석을 채운 타자만 넣음

## 타율
| 순위 | 선수 | 구단 | 타율 |
|---|---|---|---|
{top('avg')}

## 홈런
| 순위 | 선수 | 구단 | 홈런 |
|---|---|---|---|
{top('hr', fmt=lambda v: f'{v}개')}

## 타점
| 순위 | 선수 | 구단 | 타점 |
|---|---|---|---|
{top('rbi', fmt=lambda v: f'{v}점')}

## 평균자책점
| 순위 | 선수 | 구단 | 평균자책점 |
|---|---|---|---|
{top('era', rev=False, role='투수')}

## 다승
| 순위 | 선수 | 구단 | 승 |
|---|---|---|---|
{top('w', role='투수', fmt=lambda v: f'{v}승')}
""", "2026-09-27"))

    for t in B_TEAMS:
        r = next(x for x in tab if x["id"] == t[0])
        mine = [p for p in pl if p["team"] == t[0]]
        h2h = {}
        for g in b["games"]:
            if t[0] in (g["home"], g["away"]):
                me_home = g["home"] == t[0]
                op = g["away"] if me_home else g["home"]
                ms, os_ = (g["hs"], g["as"]) if me_home else (g["as"], g["hs"])
                x = h2h.setdefault(op, [0, 0, 0])
                x[0 if ms > os_ else 1 if ms < os_ else 2] += 1
        h2h_rows = "\n".join(f"| {nm(op)} | {v[0]}승 {v[1]}패 {v[2]}무 |" for op, v in sorted(h2h.items()))
        post_line = "포스트시즌에 나가지 못함"
        for sname, s in post.items():
            if t[0] in (s["hi"], s["lo"]):
                post_line = f"{sname}에서 {'이김' if s['winner'] == t[0] else '짐'}({s['score'] if s['winner'] == t[0] else s['score'][::-1]})"
        if post["한빛시리즈"]["winner"] == t[0]:
            post_line = f"한빛시리즈 우승({post['한빛시리즈']['score']})"
        players = "\n".join(
            f"| {p['name']} | {p['pos']} | " + (f"타율 {p['avg']}, 홈런 {p['hr']}개, 타점 {p['rbi']}점" if p["role"] == "타자"
                                              else f"{p['w']}승 {p['l']}패, 평균자책점 {p['era']}, 탈삼진 {p['so']}개") + " |" for p in mine)
        out.append(doc(f"team_{t[0]}", f"{t[1]} (한빛프로야구)", f"""
{eul(t[3])} 연고로 하는 한빛프로야구 구단이다. 팬들은 줄여서 '{t[2]}'{'이라고' if _batchim(t[2]) else '라고'} 부른다.

## 2026 시즌 성적
- 정규시즌 {r['rank']}위 : {r['w']}승 {r['l']}패 {r['d']}무, 승률 {r['pct_s']}, 게임차 {r['gb']}
- 득점 {r['rs']}, 실점 {r['ra']}
- 포스트시즌 : {post_line}

## 주요 선수(2026 정규시즌 기록)
| 선수 | 포지션 | 기록 |
|---|---|---|
{players}

## 상대 전적(정규시즌)
| 상대 | 전적 |
|---|---|
{h2h_rows}
""", "2026-09-27"))

    for p in pl:
        t = TEAM[p["team"]]
        if p["role"] == "타자":
            body = f"""
{t['name']} 소속 {p['pos']}이다.

## 2026 정규시즌 기록
| 경기 | 타수 | 안타 | 타율 | 홈런 | 타점 |
|---|---|---|---|---|---|
| {p['g']} | {p['ab']} | {p['h']} | {p['avg']} | {p['hr']} | {p['rbi']} |
"""
        else:
            body = f"""
{t['name']} 소속 선발투수이다.

## 2026 정규시즌 기록
| 경기 | 승 | 패 | 이닝 | 자책점 | 평균자책점 | 탈삼진 |
|---|---|---|---|---|---|---|
| {p['g']} | {p['w']} | {p['l']} | {p['ip']} | {p['er']} | {p['era']} | {p['so']} |

이닝의 소수 자리는 아웃 카운트임(160.1은 160과 1/3이닝)
"""
        out.append(doc(f"player_{p['id']}", f"{p['name']} ({t['name']})", body, "2026-09-27"))
    return out


def esports_docs(db: dict) -> list[dict]:
    e = db["esports"]
    out = []
    out.append(doc("game_chrono", "크로노 아레나(가상 게임)", """
넥서스리그와 월드 인비테이셔널의 종목인 가상 5대5 전략 게임이다. 한 세트는 보통 25~40분 걸린다.

## 포지션
| 포지션 | 하는 일 |
|---|---|
| 선봉 | 앞에서 피해를 받아 내고 싸움을 엶 |
| 유격 | 지도를 돌며 다른 길을 도움 |
| 중앙 | 가운데 길을 맡는 주력 공격수 |
| 사수 | 멀리서 지속 피해를 주는 공격수 |
| 지원 | 사수를 보호하고 시야를 잡음 |

## 기록 용어
- KDA = (킬 + 어시스트) ÷ 데스. 데스가 0이면 1로 보고 계산함. 소수 둘째 자리까지 적음
- POG 포인트 : 경기마다 가장 잘한 선수(Player of the Game)에게 100점씩 줌. 시즌 합계로 시즌 MVP를 정함
""", "2026-08-30"))
    out.append(doc("league_nxl", "넥서스리그 안내", """
넥서스리그는 10개 팀이 참가하는 가상 e스포츠 리그이다. 한 해를 스프링과 서머 두 스플릿으로 나눠 치름.

## 정규 리그
- 각 팀이 다른 9개 팀과 두 번씩, 스플릿마다 18경기를 치름. 한 경기는 3판 2선승(Bo3)
- 순위는 매치 승수로 정하고, 같으면 세트 득실(이긴 세트 - 진 세트), 그래도 같으면 팀 등록 순서로 정함

## 플레이오프
- 정규 리그 1~4위가 나감. 4강은 1위 대 4위, 2위 대 3위이며 5판 3선승(Bo5)
- 결승도 5판 3선승. 이긴 팀이 스플릿 우승팀임
- 서머 스플릿 최종 1~4위(우승, 준우승, 4강 탈락 두 팀은 정규 리그 순위대로 3·4위)가 월드 인비테이셔널에 나감
""", "2026-08-30"))
    for key, title, as_of in (("spring", "2026 넥서스리그 스프링", "2026-04-19"), ("summer", "2026 넥서스리그 서머", "2026-08-30")):
        s = e[key]
        rows = "\n".join(f"| {r['rank']} | {nm(r['id'])} | {r['mw']}승 {r['ml']}패 | {r['sw']}:{r['sl']} | {r['sd']:+d} |" for r in s["table"])
        po = s["playoffs"]
        po_rows = "\n".join(f"| {k} | {fmt_md(d(v['date']))} | {nm(v['win'])} {v['score']} {nm(v['lose'])} |" for k, v in po.items())
        fr = s["final_rank"]
        out.append(doc(f"season_nxl_{key}", title, f"""
{title} 우승팀은 **{nm(po['결승']['win'])}**이다. 결승에서 {eul(nm(po['결승']['lose']))} {ro(po['결승']['score'])} 이겼다({fmt_md(d(po['결승']['date']))}).
최종 순위는 1위 {nm(fr[0])}, 2위 {nm(fr[1])}, 3위 {nm(fr[2])}, 4위 {nm(fr[3])}이다.

## 정규 리그 순위
| 순위 | 팀 | 매치 | 세트 | 세트 득실 |
|---|---|---|---|---|
{rows}

## 플레이오프
| 경기 | 날짜 | 결과 |
|---|---|---|
{po_rows}
""", as_of))
    for t in E_TEAMS:
        sp = next(r for r in e["spring"]["table"] if r["id"] == t[0])
        su = next(r for r in e["summer"]["table"] if r["id"] == t[0])
        fsp = e["spring"]["final_rank"]
        fsu = e["summer"]["final_rank"]
        def fin(fr, tid):
            return f"최종 {fr.index(tid) + 1}위" if tid in fr else "플레이오프 탈락"
        roster = "\n".join(f"| {p['name']} | {p['pos']} | {p['k']}/{p['d']}/{p['a']} | {p['kda']} | {p['pog']} |"
                           for p in e["players"] if p["team"] == t[0])
        note = " 프로야구 구단 '청주 비룡'과 모기업이 같아 둘 다 '비룡'으로 줄여 부르기도 함" if t[0] == "e02" else ""
        out.append(doc(f"team_{t[0]}", f"{t[1]} (넥서스리그)", f"""
넥서스리그 소속 팀이다. 줄여서 '{t[2]}'{'이라고' if _batchim(t[2]) else '라고'} 부른다.{note}

## 2026 성적
| 스플릿 | 정규 리그 | 매치 | 세트 득실 | 플레이오프 |
|---|---|---|---|---|
| 스프링 | {sp['rank']}위 | {sp['mw']}승 {sp['ml']}패 | {sp['sd']:+d} | {fin(fsp, t[0])} |
| 서머 | {su['rank']}위 | {su['mw']}승 {su['ml']}패 | {su['sd']:+d} | {fin(fsu, t[0])} |

## 2026 서머 로스터와 개인 기록
킬/데스/어시스트는 서머 정규 리그 합계임

| 선수 | 포지션 | K/D/A | KDA | POG 포인트 |
|---|---|---|---|---|
{roster}
""", "2026-08-30"))
    for p in [x for x in e["players"] if x["ace"]]:
        t = TEAM[p["team"]]
        out.append(doc(f"player_{p['id']}", f"{p['name']} ({t['name']})", f"""
{t['name']}의 {p['pos']} 선수이다. 팀의 주장 역할을 맡고 있다.

## 2026 서머 정규 리그 기록
| 킬 | 데스 | 어시스트 | KDA | POG 포인트 |
|---|---|---|---|---|
| {p['k']} | {p['d']} | {p['a']} | {p['kda']} | {p['pog']} |
""", "2026-08-30"))
    return out


def wiv_docs(db: dict) -> tuple[list[dict], list[dict]]:
    w = db["wiv"]
    out = [doc("league_wiv", "월드 인비테이셔널 안내", f"""
월드 인비테이셔널은 크로노 아레나의 가상 국제 대회이다. 2026 대회에는 5개 지역 16개 팀이 나왔다.

## 참가
- 넥서스리그 4팀(서머 최종 1~4위), 북부연합·서부리그·동부리그·남부연합 각 3팀

## 방식
- 조별 리그(10월 15일~25일) : 4개 조, 조마다 4팀이 두 번씩 단판(Bo1)으로 맞붙음. 승수가 같으면 팀 등록 순서로 순위를 정함. 조 1·2위가 8강에 감
- 8강(10월 31일~11월 1일), 4강(11월 7~8일), 결승(11월 14일)은 모두 5판 3선승

## 일정 안내
- 결승은 2026년 11월 14일 예정임. 기록실 기준일({SNAPSHOT}) 현재 결승 결과는 없음
""", SNAPSHOT)]
    grp = []
    for g, rows in w["groups"].items():
        grp.append(f"## {g}조\n| 순위 | 팀 | 소속 | 전적 |\n|---|---|---|---|\n" +
                   "\n".join(f"| {r['rank']} | {nm(r['id'])} | {TEAM[r['id']]['home']} | {r['w']}승 {r['l']}패 |" for r in rows))
    out.append(doc("wiv_2026_groups", "2026 월드 인비테이셔널 조별 리그", "조 1·2위가 8강에 올랐다.\n\n" + "\n\n".join(grp), "2026-10-25"))
    ko = ["| 단계 | 날짜 | 결과 |", "|---|---|---|"]
    for k, m in enumerate(w["qf"], 1):
        ko.append(f"| 8강 {k}경기 | {fmt_md(d(m['date']))} | {nm(m['win'])} {m['score']} {nm(m['lose'])} |")
    for k, m in enumerate(w["sf"], 1):
        ko.append(f"| 4강 {k}경기 | {fmt_md(d(m['date']))} | {nm(m['win'])} {m['score']} {nm(m['lose'])} |")
    f = w["final"]
    out.append(doc("wiv_2026_knockout", "2026 월드 인비테이셔널 결선", f"""
8강부터 결선 토너먼트이다. 결승은 **{nm(w['sf'][0]['win'])} 대 {nm(w['sf'][1]['win'])}**로 11월 14일에 열릴 예정이다.

{chr(10).join(ko)}
""", SNAPSHOT))
    upd = [doc("wiv_2026_final", "2026 월드 인비테이셔널 결승 결과", f"""
11월 14일 결승에서 **{nm(f['win'])}**{'이' if _batchim(nm(f['win'])) else '가'} {eul(nm(f['lose']))} {ro(f['score'])} 꺾고 2026 월드 인비테이셔널 우승을 차지했다.
""", "2026-11-14")]
    return out, upd


def policy_docs() -> list[dict]:
    return [
        doc("policy_format", "기록실 답변 규약 1 : 필드와 표기", f"""
기록실 질문 응답은 JSON 하나로 내보냄. 필드별 규칙은 아래와 같음

## 필드
| 필드 | 규칙 |
|---|---|
| league | 한빛프로야구, 넥서스리그, 월드인비테이셔널 중 하나. 정할 수 없으면 null |
| entity | 질문의 대상인 팀·선수의 정식 이름. '누가 우승했나'처럼 팀 자체를 묻는 질문은 null로 두고 답을 value에 씀 |
| metric | 아래 '지표 이름' 표의 이름을 그대로 씀 |
| value | 표기 규약을 따른 답. 답할 수 없으면 null |
| as_of | 값이 확정된 날짜(YYYY-MM-DD). 질문에 날짜가 주어졌으면 그 날짜. 답할 수 없으면 null |
| answerable | 기록과 규약으로 답할 수 있으면 true |

## 지표 이름
순위, 승률, 게임차, 타율, 홈런, 타점, 평균자책점, 승, 탈삼진, 우승팀, 준우승팀, 시리즈 결과, 경기 결과, 세트 결과, KDA, POG 포인트, 조별 순위

## 표기 규약
| 지표 | 표기 | 예 |
|---|---|---|
| 순위·조별 순위 | n위 | 3위 |
| 승률·타율 | 소수 셋째 자리, 반올림 | 0.615 |
| 평균자책점·KDA | 소수 둘째 자리, 반올림 | 2.31 |
| 게임차 | 0.5 단위 | 3.5 |
| 홈런·탈삼진 | n개 | 31개 |
| 타점 | n점 | 95점 |
| 승 | n승 | 14승 |
| 시리즈 결과·경기 결과·세트 결과 | 물어본 팀 기준 '이긴 수:진 수' | 3:1 |
| 우승팀·준우승팀 | 정식 이름 | 노바 크라운 |

## 기준일(as_of)
- 한빛프로야구 정규시즌 기록 : 2026-09-27
- 한빛프로야구 포스트시즌 : 그 시리즈가 끝난 날
- 넥서스리그 스프링 : 2026-04-19, 서머 : 2026-08-30
- 월드 인비테이셔널 : 그 경기가 열린 날(조별 리그 순위는 2026-10-25)
- 월별 순위처럼 질문에 날짜가 있으면 그 날짜를 씀
""", SNAPSHOT),
        doc("policy_answerable", "기록실 답변 규약 2 : 답하지 않는 질문", f"""
아래에 해당하면 answerable을 false로, value와 as_of를 null로 둠. league·entity·metric은 질문에서 알 수 있는 만큼 채움

| 경우 | 예 |
|---|---|
| 기록실 기준일({SNAPSHOT}) 뒤의 결과 | 아직 열리지 않은 경기, 결승 결과 |
| 공개 경기 기록이 아닌 정보 | 연봉, 계약금, 부상 진단명, 사생활, 이적 소문 |
| 예측·평가 | 누가 우승할 것 같은지, 그 판정이 오심인지 |
| 기록실에 없는 시즌 | 2025 시즌 이전, 2027 시즌 |
| 대상을 하나로 정할 수 없음 | 두 리그에 같은 약칭이 있는데 리그를 말하지 않음 |

- 기록실은 2026 시즌 한빛프로야구, 넥서스리그, 월드 인비테이셔널 기록만 다룸
- 답할 수 없는 질문에 값을 지어내는 것은 틀린 값보다 더 나쁜 오류로 봄
""", SNAPSHOT),
        doc("policy_entity", "기록실 답변 규약 3 : 이름과 약칭", f"""
entity에는 정식 이름을 씀. 팬들이 쓰는 약칭이 들어오면 아래 표로 바꿈

| 약칭 | 정식 이름 | 리그 |
|---|---|---|
""" + "\n".join(f"| {t[2]} | {t[1]} | {LG_B} |" for t in B_TEAMS) + "\n" +
            "\n".join(f"| {t[2]} | {t[1]} | {LG_E} |" for t in E_TEAMS) + """

- '비룡'은 두 리그에 다 있음. 질문에 야구·e스포츠·리그 이름 같은 단서가 있으면 그쪽으로 정하고, 단서가 없으면 대상을 정할 수 없는 질문으로 봄
- 선수는 등록명(프로야구는 이름, e스포츠는 선수 등록 닉네임)을 씀
""", SNAPSHOT),
    ]


# ------------------------------------------------------------------ 문항
SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "S 가상 스포츠 기록실 2026 질의",
    "description": "가상 스포츠 기록실(한빛프로야구·넥서스리그·월드 인비테이셔널 2026 시즌) 질문에 답하는 JSON 하나를 만든다. 기록과 규약으로 답할 수 없으면 answerable을 false로 두고 값을 지어내지 않는다.",
    "type": "object",
    "properties": {
        "league": {"enum": [LG_B, LG_E, LG_W, None], "description": "리그·대회. 정할 수 없으면 null"},
        "entity": {"type": ["string", "null"], "description": "질문 대상 팀·선수의 정식 이름. 팀 자체를 묻는 질문(우승팀 등)이면 null"},
        "metric": {"type": ["string", "null"], "description": "묻는 지표의 표준 이름(예: 순위, 승률, 홈런, 우승팀)"},
        "value": {"type": ["string", "null"], "description": "표기 규약에 맞춘 답(예: 3위, 0.615, 31개, 3:1). 답할 수 없으면 null"},
        "as_of": {"type": ["string", "null"], "description": "값이 확정된 날짜 YYYY-MM-DD. 답할 수 없으면 null"},
        "answerable": {"type": "boolean", "description": "기록과 규약으로 답할 수 있으면 true"},
    },
    "required": ["league", "entity", "metric", "value", "as_of", "answerable"],
    "additionalProperties": False,
}

SCORING = """# S 도메인 채점 규칙: scorer.py가 읽음
# match: exact(정규화 후 일치) · enum · bool · contains(정답 문자열이 출력 안에 있음) · numseq(숫자 순서까지 같음, 숫자가 없으면 exact)
# 정답이 null 이면 출력도 null(또는 빈 값)이어야 정답
# core: true 인 필드가 모두 맞으면 Exact Match(보조 지표)
# 답 가능 여부(answerable)와 값(value)이 가중치 10 중 6: 모르는 것을 지어내지 않는 능력과 정확한 값 추출이 핵심인 추출형 과제
fields:
  league:     {match: enum,     weight: 1}
  entity:     {match: exact,    weight: 1}
  metric:     {match: contains, weight: 1}
  value:      {match: numseq,   weight: 3, core: true}
  as_of:      {match: exact,    weight: 1}
  answerable: {match: bool,     weight: 3, core: true}
"""


def item(iid, typ, text, league, entity, metric, value, as_of, answerable=True, docs=(), family=None) -> dict:
    it = {"id": iid, "type": typ, "input": text,
          "gold": {"league": league, "entity": entity, "metric": metric, "value": value, "as_of": as_of, "answerable": answerable},
          "gold_doc_ids": list(docs)}
    if family:
        it["family"] = family
    return it


class Maker:
    """문항 틀. split='eval'이면 평가용 대상·문장 틀, 'train'이면 학습용 대상·문장 틀을 씀"""

    def __init__(self, db: dict, split: str, rng: random.Random):
        self.db, self.split, self.rng = db, split, rng
        b, e = db["baseball"], db["esports"]
        self.bteams = [t[0] for t in B_TEAMS if t[5] == split]
        self.eteams = [t[0] for t in E_TEAMS if t[5] == split]
        self.bpl = [p for p in b["players"] if p["use"] == split]
        self.epl = [p for p in e["players"] if p["use"] == split]
        self.eace = [p for p in self.epl if p["ace"]]

    def pick(self, xs):
        return self.rng.choice(xs)

    def say(self, ev: list[str], tr: list[str]) -> str:
        return self.pick(ev if self.split == "eval" else tr)

    # ---------- K : 코퍼스를 봐야 답이 나옴
    def k_rank(self):
        t = self.pick(self.bteams)
        r = next(x for x in self.db["baseball"]["table"] if x["id"] == t)
        q = self.say([f"{nm(t)} 올해 정규시즌 최종 몇 위로 끝났나요?", f"{TEAM[t]['short']} 이번 시즌 정규리그 마지막 순위가 궁금해요. 야구요"],
                     [f"{nm(t)} 2026 정규시즌 최종 순위 알려줘", f"야구 {TEAM[t]['short']} 올해 몇 등으로 시즌 마쳤어?"])
        return q, (LG_B, nm(t), "순위", f"{r['rank']}위", "2026-09-27", True, ["season_hpb_2026", f"team_{t}"]), "b_rank"

    def k_month(self):
        t = self.pick(self.bteams)
        m = self.pick(list(self.db["baseball"]["monthly"]))
        r = next(x for x in self.db["baseball"]["monthly"][m] if x["id"] == t)
        mm = d(m).month
        q = self.say([f"{mm}월 말 기준으로 {eun(nm(t))} 몇 위였어요?", f"한빛프로야구 {mm}월 끝났을 때 {nm(t)} 순위 좀 알려 주세요"],
                     [f"{nm(t)} {mm}월 마지막 날 순위가 어땠지?", f"{mm}월 말 시점 {nm(t)} 순위"])
        return q, (LG_B, nm(t), "순위", f"{r['rank']}위", m, True, ["season_hpb_2026"]), "b_month"

    def k_pct(self):
        t = self.pick(self.bteams)
        r = next(x for x in self.db["baseball"]["table"] if x["id"] == t)
        q = self.say([f"{nm(t)} 정규시즌 승률이 얼마였나요?", f"{nm(t)} 올 시즌 승률 몇이에요?"],
                     [f"{nm(t)} 2026 정규시즌 승률은?", f"{TEAM[t]['short']} 야구팀 시즌 승률 알려줘"])
        return q, (LG_B, nm(t), "승률", r["pct_s"], "2026-09-27", True, ["season_hpb_2026", f"team_{t}"]), "b_pct"

    def k_player(self):
        p = self.pick([x for x in self.bpl])
        if p["role"] == "타자":
            met, val = self.pick([("홈런", f"{p['hr']}개"), ("타율", p["avg"]), ("타점", f"{p['rbi']}점")])
        else:
            met, val = self.pick([("평균자책점", p["era"]), ("승", f"{p['w']}승"), ("탈삼진", f"{p['so']}개")])
        q = self.say([f"{p['name']} 선수 올 시즌 {met} 기록이 어떻게 돼요?", f"{nm(p['team'])} {p['name']} {met} 몇이었는지 알려 주세요"],
                     [f"{p['name']} 2026 정규시즌 {met}은?", f"{p['name']}({nm(p['team'])}) {met} 기록 궁금"])
        return q, (LG_B, p["name"], met, val, "2026-09-27", True, [f"player_{p['id']}", f"team_{p['team']}"]), "b_player"

    def k_post(self):
        post = self.db["baseball"]["post"]
        sname = self.pick(list(post))
        s = post[sname]
        if self.rng.random() < 0.5:
            q = self.say([f"올해 야구 {sname}에서 누가 이겼어요?", f"2026 한빛프로야구 {sname} 승리 팀이 어디죠?"],
                         [f"{sname} 이긴 팀 알려줘(한빛프로야구)", f"야구 {sname} 결과 누가 올라갔어?"])
            met = "우승팀" if sname == "한빛시리즈" else "시리즈 결과"
            if met == "우승팀":
                return q, (LG_B, None, "우승팀", nm(s["winner"]), s["end"], True, ["post_hpb_2026"]), "b_post"
            return q, (LG_B, nm(s["winner"]), "시리즈 결과", s["score"], s["end"], True, ["post_hpb_2026"]), "b_post"
        t = s["loser"]
        q = self.say([f"{eun(nm(t))} {sname}에서 몇 대 몇으로 졌나요?", f"{sname} {nm(t)} 시리즈 전적이 어떻게 됐죠?"],
                     [f"{nm(t)} {sname} 시리즈 결과", f"{sname}에서 {nm(t)} 몇 승 몇 패였어?"])
        return q, (LG_B, nm(t), "시리즈 결과", s["score"][::-1], s["end"], True, ["post_hpb_2026"]), "b_post"

    def k_nxl(self):
        key = self.pick(["spring", "summer"])
        s = self.db["esports"][key]
        kname = "스프링" if key == "spring" else "서머"
        as_of = "2026-04-19" if key == "spring" else "2026-08-30"
        c = self.rng.random()
        if c < 0.35:
            q = self.say([f"넥서스리그 {kname} 우승팀이 어디예요?", f"2026 넥서스리그 {kname} 누가 우승했어요?"],
                         [f"{kname} 넥서스리그 우승한 팀", f"올해 넥서스리그 {kname} 챔피언은?"])
            return q, (LG_E, None, "우승팀", nm(s["playoffs"]["결승"]["win"]), as_of, True, [f"season_nxl_{key}"]), "e_champ"
        t = self.pick(self.eteams)
        r = next(x for x in s["table"] if x["id"] == t)
        if c < 0.7:
            q = self.say([f"{nm(t)} 넥서스리그 {kname} 정규 리그 몇 위였죠?", f"{kname} 정규 리그에서 {nm(t)} 순위 알려 주세요"],
                         [f"{nm(t)} {kname} 정규 리그 순위", f"넥서스리그 {kname} {nm(t)} 몇 위?"])
            return q, (LG_E, nm(t), "순위", f"{r['rank']}위", as_of, True, [f"season_nxl_{key}", f"team_{t}"]), "e_rank"
        p = self.pick([x for x in self.epl if x["team"] == t])
        q = self.say([f"{nm(t)} {p['name']} 선수 서머 KDA가 몇이에요?", f"{p['name']} 선수 이번 서머 KDA 알려 주세요"],
                     [f"{p['name']}({nm(t)}) 서머 KDA", f"{nm(t)} {p['pos']} {p['name']} KDA는?"])
        return q, (LG_E, p["name"], "KDA", p["kda"], "2026-08-30", True, [f"team_{t}"] + ([f"player_{p['id']}"] if p["ace"] else [])), "e_kda"

    def k_wiv(self):
        w = self.db["wiv"]
        mine = [r for g in w["groups"].values() for r in g if r["id"] in self.eteams + [x[0] for x in W_FOREIGN]]
        r = self.pick(mine)
        g = next(k for k, rows in w["groups"].items() if r in rows)
        if self.rng.random() < 0.5:
            q = self.say([f"월드 인비테이셔널 {g}조에서 {eun(nm(r['id']))} 몇 위였나요?", f"{nm(r['id'])} 이번 월드 인비테이셔널 조별 리그 순위가 궁금해요"],
                         [f"월드 인비테이셔널 조별 리그 {nm(r['id'])} 순위", f"{g}조 {nm(r['id'])} 몇 위로 끝났어?"])
            return q, (LG_W, nm(r["id"]), "조별 순위", f"{r['rank']}위", "2026-10-25", True, ["wiv_2026_groups"]), "w_group"
        m = self.pick(w["qf"] + w["sf"])
        stage = "8강" if m in w["qf"] else "4강"
        t = m["lose"]
        q = self.say([f"월드 인비테이셔널 {stage}에서 {nm(t)} 경기 결과가 어떻게 됐어요?", f"{nm(t)} 월드 인비테이셔널 {stage} 몇 대 몇이었죠?"],
                     [f"{nm(t)} {stage} 경기 결과(월드 인비테이셔널)", f"월드 인비테이셔널 {stage} {nm(t)} 세트 스코어"])
        return q, (LG_W, nm(t), "경기 결과", m["score"][::-1], m["date"], True, ["wiv_2026_knockout"]), "w_ko"

    # ---------- R : 입력 안에 근거가 다 있음(계산·추출)
    def r_winrate(self):
        g = self.rng.randint(50, 66)
        dd = self.rng.randint(0, 2)
        w = round((g - dd) * self.rng.uniform(0.36, 0.64))
        l = g - dd - w
        t = self.pick(self.bteams)
        day = self.rng.randint(5, 25)
        q = self.say([f"한빛프로야구 {iga(nm(t))} 7월 {day}일까지 {w}승 {l}패 {dd}무예요. 이때 승률은 얼마죠?",
                      f"7월 {day}일 기준 {nm(t)} 성적 {w}승 {dd}무 {l}패. 승률 계산해 주세요"],
                     [f"{nm(t)} 7월 {day}일 현재 {w}승 {l}패 {dd}무, 승률은?", f"승 {w} 패 {l} 무 {dd}일 때 {nm(t)} 승률(7월 {day}일 기준)"])
        return q, (LG_B, nm(t), "승률", R.winrate(w, l), f"2026-07-{day:02d}", True, []), "r_winrate"

    def r_gb(self):
        t1, t2 = self.rng.sample(self.bteams, 2)
        w1, l1 = self.rng.randint(38, 46), self.rng.randint(22, 30)
        w2, l2 = w1 - self.rng.randint(1, 8), l1 + self.rng.randint(0, 6)
        q = self.say([f"8월 20일 현재 1위 {eun(nm(t1))} {w1}승 {l1}패, {eun(nm(t2))} {w2}승 {l2}패예요. {nm(t2)}의 게임차는 얼마죠?",
                      f"야구 순위표 보니 {nm(t1)} {w1}-{l1}, {nm(t2)} {w2}-{l2}이던데(8월 20일) {iga(nm(t2))} 1위와 몇 게임차예요?"],
                     [f"1위 {nm(t1)} {w1}승 {l1}패, {nm(t2)} {w2}승 {l2}패(8월 20일). {nm(t2)} 게임차?", f"{nm(t2)} {w2}승 {l2}패, 선두 {nm(t1)} {w1}승 {l1}패일 때 게임차 계산(8월 20일)"])
        return q, (LG_B, nm(t2), "게임차", R.games_behind((w1, l1), (w2, l2)), "2026-08-20", True, []), "r_gb"

    def r_avg(self):
        p = self.pick([x for x in self.bpl if x["role"] == "타자"])
        ab = self.rng.randint(80, 300)
        h = round(ab * self.rng.uniform(0.21, 0.36))
        q = self.say([f"{nm(p['team'])} {p['name']} 선수가 6월 30일까지 {ab}타수 {h}안타래요. 그때 타율이 얼마죠?",
                      f"6월 30일 기준 {p['name']} {h}안타 / {ab}타수. 타율은요?"],
                     [f"{p['name']} {ab}타수 {h}안타(6월 30일). 타율?", f"6월 30일까지 {p['name']} 안타 {h} 타수 {ab}, 타율 계산"])
        return q, (LG_B, p["name"], "타율", R.batting_avg(h, ab), "2026-06-30", True, []), "r_avg"

    def r_era(self):
        p = self.pick([x for x in self.bpl if x["role"] == "투수"])
        outs = self.rng.randint(90, 330)
        ip = f"{outs // 3}.{outs % 3}"
        er = round(outs / 27 * self.rng.uniform(2.0, 5.5))
        q = self.say([f"{p['name']} 투수가 5월 31일까지 {ip}이닝 동안 자책점 {er}점을 내줬어요. 평균자책점은 얼마인가요?(이닝 소수 자리는 아웃 카운트)",
                      f"5월 말 기준 {p['name']} {ip}이닝 {er}자책. 평균자책점 계산해 주세요. 이닝의 .1은 1/3이닝이에요"],
                     [f"{p['name']} {ip}이닝 자책 {er}(5월 31일). 평균자책점은? 이닝 소수 자리는 아웃 수", f"5월 31일까지 {p['name']} 자책점 {er}, 이닝 {ip}(소수 자리=아웃). 평균자책점"])
        return q, (LG_B, p["name"], "평균자책점", R.era(er, ip), "2026-05-31", True, []), "r_era"

    def r_series(self):
        t = self.pick(self.bteams)
        n = self.rng.choice([3, 4, 5])
        sc = []
        for _ in range(n):
            a, b = self.rng.randint(0, 9), self.rng.randint(0, 9)
            if a == b:
                a += 1
            sc.append(f"{a}:{b}")
        txt = ", ".join(f"{k}차전 {s}" for k, s in enumerate(sc, 1))
        q = self.say([f"{nm(t)} 연습 시리즈 결과가 우리 팀 점수 먼저 {txt}였어요. 시리즈 결과를 정리하면요?",
                      f"우리 {nm(t)} 점수 먼저 적으면 {txt}. 몇 승 몇 패인가요?"],
                     [f"{nm(t)} 기준 {txt}. 시리즈 결과는?", f"{txt}({nm(t)} 점수 먼저) 이긴 수:진 수"])
        return q, (LG_B, nm(t), "시리즈 결과", R.series(sc), None, True, []), "r_series"

    def r_kda(self):
        p = self.pick(self.epl)
        k, dd, a = self.rng.randint(0, 15), self.rng.randint(0, 9), self.rng.randint(0, 25)
        q = self.say([f"넥서스리그 경기에서 {p['name']} 선수가 {k}킬 {dd}데스 {a}어시스트를 했어요. 이 경기 KDA는요?",
                      f"{nm(p['team'])} {p['name']} 오늘 경기 {k}/{dd}/{a}였는데 KDA 얼마예요?"],
                     [f"{p['name']} {k}/{dd}/{a}, KDA 계산", f"넥서스리그 {p['name']} 킬 {k} 데스 {dd} 어시 {a}일 때 KDA"])
        return q, (LG_E, p["name"], "KDA", R.kda(k, dd, a), None, True, []), "r_kda"

    def r_sets(self):
        t = self.pick(self.eteams)
        n = self.rng.choice([3, 4, 5])
        res = []
        while True:
            res = [self.rng.choice(["승", "패"]) for _ in range(n)]
            if res.count("승") != res.count("패"):
                break
        txt = ", ".join(f"{k}세트 {r}" for k, r in enumerate(res, 1))
        q = self.say([f"넥서스리그 {nm(t)} 오늘 경기 {txt}였어요. 세트 결과를 {nm(t)} 기준으로 적으면요?",
                      f"{nm(t)} 경기 세트별로 {txt}. 세트 스코어가 몇 대 몇이죠?"],
                     [f"{nm(t)} {txt}. 세트 결과", f"{txt}({nm(t)} 기준) 세트 스코어"])
        return q, (LG_E, nm(t), "세트 결과", f"{res.count('승')}:{res.count('패')}", None, True, []), "r_sets"

    # ---------- F : 규약을 알아야 맞힘(답하지 않는 질문 판단)
    def f_future(self):
        q = self.say(["월드 인비테이셔널 2026 우승팀 어디예요?", "올해 월드 인비테이셔널 결승 몇 대 몇으로 끝났어요?"],
                     ["2026 월드 인비테이셔널 최종 우승팀은?", "월드 인비테이셔널 결승 결과 알려줘"])
        met = "경기 결과" if "몇 대 몇" in q or "결과" in q else "우승팀"
        return q, (LG_W, None, met, None, None, False, ["league_wiv", "policy_answerable"]), "f_future"

    def f_private(self):
        p = self.pick(self.bpl + self.eace)
        lg = LG_B if p["team"].startswith("b") else LG_E
        met = self.pick(["연봉", "계약금", "부상", "이적"])
        q = self.say([f"{p['name']} 선수 {met} 얘기 들었는데 사실이에요? 얼마예요?", f"{nm(p['team'])} {p['name']} {met} 관련해서 정확히 알려 주세요"],
                     [f"{p['name']} {met} 정보", f"{p['name']} {met} 어떻게 된 거야?"])
        return q, (lg, p["name"], met, None, None, False, ["policy_answerable"]), "f_private"

    def f_predict(self):
        if self.rng.random() < 0.5:
            t = self.pick(self.bteams)
            q = self.say([f"{nm(t)} 내년에 우승할 수 있을까요?", f"{nm(t)} 다음 시즌 몇 위 할 것 같아요?"],
                         [f"{nm(t)} 2027 시즌 예상 순위", f"{nm(t)} 내년 우승 가능성"])
            met = "우승팀" if "우승" in q else "순위"
            return q, (LG_B, nm(t), met, None, None, False, ["policy_answerable"]), "f_predict"
        t = self.pick(self.eteams)
        q = self.say([f"넥서스리그 {nm(t)} 지난 경기 3세트 판정 오심 맞죠?", f"{nm(t)} 오늘 경기 심판 판정 공정했나요?"],
                     [f"{nm(t)} 경기 판정 오심인지 판단해줘", f"{nm(t)} 판정 논란 정리하면 누가 맞아?"])
        return q, (LG_E, nm(t), "경기 결과", None, None, False, ["policy_answerable"]), "f_predict"

    def f_season(self):
        t = self.pick(self.bteams)
        yr = self.pick(["2025", "2024"])
        met, txt = self.pick([("순위", "몇 위였어요"), ("승률", "승률이 얼마였죠"), ("우승팀", "우승팀이 어디였나요")])
        if met == "우승팀":
            q = self.say([f"{yr} 시즌 한빛프로야구 {txt}?", f"{yr}년 한빛시리즈 {txt}?"], [f"{yr} 한빛프로야구 우승팀", f"{yr}년 야구 우승한 팀"])
            return q, (LG_B, None, "우승팀", None, None, False, ["policy_answerable"]), "f_season"
        q = self.say([f"{nm(t)} {yr} 시즌에는 {txt}?", f"{yr}년 {nm(t)} 정규시즌 {txt}?"], [f"{nm(t)} {yr} {met}", f"{yr} 시즌 {nm(t)} {met} 알려줘"])
        return q, (LG_B, nm(t), met, None, None, False, ["policy_answerable"]), "f_season"

    def f_ambiguous(self):
        met, txt = self.pick([("순위", "이번 시즌 몇 위였어요"), ("우승팀", "올해 우승했어요")])
        if met == "우승팀":
            met = "순위"
        q = self.say([f"비룡 {txt}?", "비룡 올해 성적 몇 위로 끝났는지 알려 주세요"], ["비룡 최종 순위", "비룡 시즌 몇 등?"])
        return q, (None, None, met, None, None, False, ["policy_entity", "policy_answerable"]), "f_ambiguous"

    def f_short_ok(self):
        """약칭이지만 단서가 있어 답할 수 있는 경우(판단형의 짝). 학습용은 평가용과 다른 지표를 물음"""
        if self.split == "train":
            if self.rng.random() < 0.5:
                r = next(x for x in self.db["baseball"]["table"] if x["id"] == "b01")
                q = self.pick(["야구 비룡 정규시즌 승률 얼마야?", "한빛프로야구 비룡 시즌 승률"])
                return q, (LG_B, nm("b01"), "승률", r["pct_s"], "2026-09-27", True, ["policy_entity"]), "f_short_ok"
            r = next(x for x in self.db["esports"]["spring"]["table"] if x["id"] == "e02")
            q = self.pick(["넥서스리그 비룡 스프링 정규 리그 순위는?", "게임단 비룡 스프링 몇 위였지?"])
            return q, (LG_E, nm("e02"), "순위", f"{r['rank']}위", "2026-04-19", True, ["policy_entity"]), "f_short_ok"
        if self.rng.random() < 0.5:
            r = next(x for x in self.db["baseball"]["table"] if x["id"] == "b01")
            q = self.say(["야구 비룡 이번 시즌 몇 위로 끝났어요?", "프로야구 쪽 비룡 정규시즌 최종 순위요"], ["한빛프로야구 비룡 최종 순위", "야구팀 비룡 몇 위?"])
            return q, (LG_B, nm("b01"), "순위", f"{r['rank']}위", "2026-09-27", True, ["policy_entity", "season_hpb_2026"]), "f_short_ok"
        r = next(x for x in self.db["esports"]["summer"]["table"] if x["id"] == "e02")
        q = self.say(["넥서스리그 비룡 서머 정규 리그 몇 위였죠?", "e스포츠 비룡 서머 정규 리그 순위 알려 주세요"], ["넥서스리그 비룡 서머 순위", "게임단 비룡 서머 몇 위?"])
        return q, (LG_E, nm("e02"), "순위", f"{r['rank']}위", "2026-08-30", True, ["policy_entity", "season_nxl_summer"]), "f_short_ok"


K_FNS = ["k_rank", "k_month", "k_pct", "k_player", "k_post", "k_nxl", "k_wiv"]
R_FNS = ["r_winrate", "r_gb", "r_avg", "r_era", "r_series", "r_kda", "r_sets"]
F_FNS = ["f_future", "f_private", "f_predict", "f_season", "f_ambiguous", "f_short_ok"]


def build_items(mk: Maker, prefix: str, plan: list[tuple[str, str]], seen: set[str]) -> list[dict]:
    out = []
    for k, (typ, fn) in enumerate(plan, 1):
        for _ in range(60):
            q, g, fam = getattr(mk, fn)()
            if q not in seen:
                break
        seen.add(q)
        out.append(item(f"{prefix}{k:02d}", typ, q, g[0], g[1], g[2], g[3], g[4], g[5], g[6], fam))
    return out


def main() -> None:
    db = build_db()
    (HERE / "sports_db.json").write_text(json.dumps(db, ensure_ascii=False, indent=1), encoding="utf-8")
    docs = baseball_docs(db) + esports_docs(db)
    w_docs, upd = wiv_docs(db)
    docs += w_docs + policy_docs()
    (DATA / "corpus").mkdir(exist_ok=True)
    for f in (DATA / "corpus").glob("*.md"):
        f.unlink()
    manifest = []
    for x in docs:
        (DATA / "corpus" / f"{x['doc_id']}.md").write_text(x["md"], encoding="utf-8")
        manifest.append({"doc_id": x["doc_id"], "title": x["title"], "file": f"corpus/{x['doc_id']}.md", "source": SOURCE,
                         "license": LICENSE, "as_of": x["as_of"], "chars": len(x["md"])})
    (DATA / "corpus_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    (DATA / "corpus_updates").mkdir(exist_ok=True)
    um = []
    for x in upd:
        (DATA / "corpus_updates" / f"{x['doc_id']}.md").write_text(x["md"], encoding="utf-8")
        um.append({"doc_id": x["doc_id"], "title": x["title"], "file": f"corpus_updates/{x['doc_id']}.md", "source": SOURCE,
                   "license": LICENSE, "as_of": x["as_of"], "chars": len(x["md"])})
    (DATA / "corpus_updates" / "updates_manifest.json").write_text(json.dumps(um, ensure_ascii=False, indent=1), encoding="utf-8")
    (DATA / "schema.json").write_text(json.dumps(SCHEMA, ensure_ascii=False, indent=2), encoding="utf-8")
    (DATA / "scoring.yaml").write_text(SCORING, encoding="utf-8")

    seen: set[str] = set()
    rng = random.Random(db["seed"] * 31 + 1)
    ev = Maker(db, "eval", rng)
    seed_plan = [("K", "k_rank"), ("K", "k_player"), ("K", "k_post"), ("K", "k_nxl"),
                 ("R", "r_winrate"), ("R", "r_series"), ("R", "r_kda"),
                 ("F", "f_future"), ("F", "f_private"), ("F", "f_ambiguous")]
    seed = build_items(ev, "S-S-", seed_plan, seen)
    for p in ("test_seed.json", "test_dataset.json"):
        (DATA / p).write_text(json.dumps(seed, ensure_ascii=False, indent=1), encoding="utf-8")

    pool_plan = [("K", f) for f in K_FNS for _ in range(2)] + [("K", "k_player")] + \
                [("R", f) for f in R_FNS for _ in range(2)] + [("R", "r_winrate")] + \
                [("F", f) for f in F_FNS for _ in range(2)] + [("F", "f_private"), ("F", "f_season"), ("F", "f_short_ok")]
    pool = build_items(ev, "S-P-", pool_plan, seen)
    for k, it in enumerate(pool, 1):
        it["id"] = f"S-P-{it['type']}{k:02d}"
    SOL.mkdir(parents=True, exist_ok=True)
    (SOL / "test_pool.json").write_text(json.dumps(pool, ensure_ascii=False, indent=1), encoding="utf-8")

    tmpl = {"domain": "S", "allocation": {"K": 8, "R": 7, "F": 5},
            "rules": ["K(지식형): 코퍼스(corpus/*.md)의 특정 문서를 봐야만 답할 수 있는 질문. gold_doc_ids에 근거 문서 id를 반드시 적는다",
                      "R(추론형): 질문 문장 안에 계산·추출 근거가 모두 있는 질문. 정답 수치는 build/s_rules.py로 계산한다",
                      "F(형식·판단형): 기록실 답변 규약(policy_*)을 알아야 맞힐 수 있는 질문. 답하지 않는 질문과, 약칭이지만 단서가 있어 답할 수 있는 질문을 섞는다",
                      "gold는 schema.json을 통과해야 한다. 값이 없으면 null. 표기는 policy_format 문서를 따른다",
                      "시드 문항과 같은 문장·같은 사실을 다시 쓰지 않는다(표현만 바꾼 문항 금지)",
                      "실존 구단·선수·대회 이름을 넣지 않는다. 모두 가상 리그 기록에서 고른다"],
            "example": seed[0],
            "items": [{"id": f"S-T-{k:02d}", "type": t, "input": "",
                       "gold": {"league": None, "entity": None, "metric": None, "value": None, "as_of": None, "answerable": True},
                       "gold_doc_ids": []} for k, t in enumerate(["K"] * 8 + ["R"] * 7 + ["F"] * 5, 1)]}
    (DATA / "team_item_template.json").write_text(json.dumps(tmpl, ensure_ascii=False, indent=1), encoding="utf-8")

    tr = Maker(db, "train", random.Random(db["seed"] * 31 + 2))
    train_plan = [("K", f) for f in K_FNS for _ in range(10)] + [("R", f) for f in R_FNS for _ in range(8)] + \
                 [("F", f) for f in F_FNS for _ in range(8)]
    train = build_items(tr, "S-L-", train_plan, seen)
    for k, it in enumerate(train, 1):
        it["id"] = f"S-L-{k:03d}"
        it.pop("family", None)
        it.pop("gold_doc_ids", None)
    rng2 = random.Random(5)
    rng2.shuffle(train)
    with open(DATA / "train_seed.jsonl", "w", encoding="utf-8") as f:
        for it in train:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    (DATA / "tools_spec.md").write_text(TOOLS_SPEC, encoding="utf-8")
    print(f"시드 {db['seed']} · 문서 {len(manifest)}건(+갱신 {len(um)}) · 시험 시드 {len(seed)} · 교체 풀 {len(pool)} · 학습 시드 {len(train)}")


TOOLS_SPEC = """# S 도메인 Tool 후보 (8~10회차 인계용)

> 기준일 2026-11-08. 시그니처는 제안이며 차승희 박사 회차에서 확정함. 데이터 원본은 `build/sports_db.json`

| Tool | 입력 | 출력 | 용도 |
|---|---|---|---|
| `standings(league, as_of)` | 리그, 날짜 | 그날 기준 순위표 | 월별 순위·순위 질문(K) 확인 |
| `player_stats(name)` | 선수 등록명 | 2026 시즌 기록 JSON | 개인 기록 질문(K) 확인 |
| `calc(metric, **args)` | 승률·게임차·타율·평균자책점·KDA와 재료 수치 | 규약 표기 값 | 계산 질문(R)에서 LLM 산술 오류 보완(`build/s_rules.py`) |
| `normalize_entity(text)` | 약칭이 섞인 질문 | 정식 이름 후보와 리그 | 약칭·동명 판단(F) |
| `schedule(league, date)` | 리그, 날짜 | 경기 일정·결과 | 기준일 뒤 결과 여부 판단 |
"""

if __name__ == "__main__":
    main()

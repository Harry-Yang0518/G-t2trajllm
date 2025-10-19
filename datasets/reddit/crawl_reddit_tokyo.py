import os, re, csv, json, time, random, hashlib
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode
import requests
from bs4 import BeautifulSoup
from dateutil import parser as dateparser

# -----------------------
# Config
# -----------------------
# Reddit-only crawl (r/JapanTravel primary). You can extend to ["JapanTravel","travel","solotravel"]
SUBREDDITS = ["JapanTravel"]

# Bias toward *post-hoc* reports, not planning/check posts
QUERY_TERMS = [
    'flair_name:"Trip Report" tokyo',
    'trip report tokyo',
    'trip review tokyo',
    'travelogue tokyo',
    '"just got back" tokyo',
    '"days in tokyo" trip report',
    'tokyo journal',
    'tokyo travel report'
]

TOKYO_TERMS = [
    'tokyo','東京','渋谷','新宿','浅草','上野','秋葉原','お台場','銀座','原宿','六本木','池袋'
]

# How many total posts to try to save (after filtering)
TARGET_POSTS = 50

# Time window: last N years
YEARS_BACK = 3

OUT_JSONL = "reddit_tokyo_tripreports.jsonl"
OUT_CSV   = "reddit_tokyo_tripreports.csv"

USER_AGENT_FALLBACK = "CapstoneCrawler/0.5 (no-auth) https://github.com"
REQUEST_TIMEOUT = 20
SLEEP_BETWEEN = (0.8, 1.6)  # polite sleeps for fallback http

# Heuristics to avoid "thin" posts that are likely not proper trip reports
MIN_TEXT_LEN = 500
MIN_SCORE    = 1

# -----------------------
# Helpers
# -----------------------
def h(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:24]

def now_utc():
    return datetime.now(timezone.utc)

def terminal_print(idx, title, url, date_str, text):
    print("\n" + "="*90)
    print(f"[{idx}] {title}")
    print(f"URL: {url}")
    print(f"Date: {date_str}")
    print((text[:500].replace("\n"," ") + (" ..." if len(text) > 500 else "")) if text else "(no body)")
    print("="*90 + "\n")

def looks_tokyoish(title, body):
    blob = f"{title or ''}\n{body or ''}".lower()
    return any(t in blob for t in TOKYO_TERMS)

NEGATIVE_TERMS = [
    "itinerary check", "check my itinerary", "is this itinerary",
    "feedback on itinerary", "rate my itinerary", "planning help",
    "draft itinerary", "is this okay", "pls review", "please review",
    "optimize my itinerary", "advise", "validate", "feedback request",
    "help plan", "planning advice", "suggestions for itinerary"
]

def looks_tripreport(title, body):
    """Prefer true trip reports / journals; reject planning/check posts."""
    blob = f"{title or ''}\n{body or ''}".lower()
    if any(t in blob for t in NEGATIVE_TERMS):
        return False
    # strong positive signals
    positive_regexes = [
        r"\btrip report\b", r"\btrip review\b", r"\btravelogue\b", r"\bjournal\b",
        r"\bjust got back\b",
        r"\bday\s*1\b", r"\bday\s*2\b", r"\bdays?\s+in\s+tokyo\b",
        r"\b(itinerary)\b.*\b(recap|lessons|what i did|what we did)\b"
    ]
    if any(re.search(p, blob) for p in positive_regexes):
        return True
    # fallback: long self-posts with “trip” + “tokyo”
    return ("trip" in blob and "tokyo" in blob and len(blob) > 600)

def clean_markdown(md):
    # light cleanup of Reddit markdown/HTML
    if not md: return ""
    # remove link formatting
    md = re.sub(r'\[(.*?)\]\((.*?)\)', r'\1', md)
    # remove inline/blocked code
    md = re.sub(r'`{1,3}.*?`{1,3}', ' ', md, flags=re.S)
    # remove blockquotes
    md = re.sub(r'>\s?', '', md)
    # collapse whitespace
    md = re.sub(r'\s+', ' ', md)
    return md.strip()

def dt_to_datestr(dt):
    if not dt: return ""
    if isinstance(dt, (int, float)):  # epoch
        return datetime.utcfromtimestamp(dt).date().isoformat()
    try:
        return dateparser.parse(str(dt)).date().isoformat()
    except Exception:
        return ""

def save_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def save_csv(records, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id","title","url","date","subreddit","author","score","num_comments","text"])
        for r in records:
            w.writerow([
                r.get("id",""),
                r.get("title",""),
                r.get("url",""),
                r.get("date",""),
                r.get("subreddit",""),
                r.get("author",""),
                r.get("score",""),
                r.get("num_comments",""),
                (r.get("text","") or "").replace("\n"," ")
            ])

# -----------------------
# Path A: PRAW (auth)
# -----------------------
def use_praw():
    cid  = os.getenv("REDDIT_CLIENT_ID")
    csec = os.getenv("REDDIT_CLIENT_SECRET")
    ua   = os.getenv("REDDIT_USER_AGENT")
    return bool(cid and csec and ua)

def crawl_with_praw():
    import praw
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT"),
        ratelimit_seconds=5
    )
    records = []
    since = now_utc() - timedelta(days=365*YEARS_BACK)

    for sub in SUBREDDITS:
        sr = reddit.subreddit(sub)
        for q in QUERY_TERMS:
            # Reddit search caps; merge/dedup later
            for post in sr.search(q, sort="new", limit=200, time_filter="all"):
                created = datetime.fromtimestamp(post.created_utc, tz=timezone.utc) if getattr(post, "created_utc", None) else None
                if created and created < since:
                    continue

                title = post.title or ""
                body  = (post.selftext or "").strip()
                if not (looks_tokyoish(title, body) and looks_tripreport(title, body)):
                    continue

                text = clean_markdown(body)
                url  = "https://www.reddit.com" + post.permalink
                score = int(post.score) if post.score is not None else 0
                if len(text) < MIN_TEXT_LEN or score < MIN_SCORE:
                    continue

                rec = {
                    "id": h(url),
                    "title": title,
                    "url": url,
                    "date": dt_to_datestr(post.created_utc),
                    "subreddit": f"r/{sub}",
                    "author": str(post.author) if post.author else "",
                    "score": score,
                    "num_comments": int(post.num_comments) if post.num_comments is not None else 0,
                    "text": text
                }
                records.append(rec)
                if len(records) >= TARGET_POSTS:
                    return dedup(records)
    return dedup(records)

# -----------------------
# Path B: Public JSON (no-auth fallback)
# -----------------------
def reddit_search_fallback(sub, query, after=None):
    # Unofficial public JSON endpoint (works for many queries without OAuth)
    base = f"https://www.reddit.com/r/{sub}/search.json"
    params = {
        "q": query,
        "restrict_sr": "1",
        "sort": "new",
        "t": "all",
        "limit": "100"
    }
    if after:
        params["after"] = after
    headers = {"User-Agent": USER_AGENT_FALLBACK}
    r = requests.get(base, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
    if r.status_code != 200:
        return None
    return r.json()

def crawl_with_fallback():
    records = []
    since = now_utc() - timedelta(days=365*YEARS_BACK)

    for sub in SUBREDDITS:
        for q in QUERY_TERMS:
            after = None
            tries = 0
            while tries < 8:  # a few pages per query
                tries += 1
                data = reddit_search_fallback(sub, q, after=after)
                if not data or "data" not in data:
                    break
                posts = data["data"].get("children", [])
                after = data["data"].get("after")
                for c in posts:
                    d = c.get("data", {})
                    title = d.get("title","")
                    body  = d.get("selftext","") or ""
                    created_utc = d.get("created_utc")
                    created = None
                    if created_utc:
                        created = datetime.fromtimestamp(created_utc, tz=timezone.utc)
                        if created < since:
                            continue
                    if not (looks_tokyoish(title, body) and looks_tripreport(title, body)):
                        continue
                    text = clean_markdown(body)
                    url  = "https://www.reddit.com" + d.get("permalink","")
                    score = int(d.get("score") or 0)
                    if len(text) < MIN_TEXT_LEN or score < MIN_SCORE:
                        continue
                    rec = {
                        "id": h(url or title),
                        "title": title,
                        "url": url,
                        "date": dt_to_datestr(created_utc),
                        "subreddit": f"r/{sub}",
                        "author": d.get("author",""),
                        "score": score,
                        "num_comments": int(d.get("num_comments") or 0),
                        "text": text
                    }
                    records.append(rec)
                    if len(records) >= TARGET_POSTS:
                        return dedup(records)

                time.sleep(random.uniform(*SLEEP_BETWEEN))
                if not after:
                    break
    return dedup(records)

def dedup(records):
    seen = set()
    out = []
    for r in records:
        k = (r.get("url") or r.get("title"))
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out

# -----------------------
# Main
# -----------------------
def main():
    if os.path.exists(OUT_JSONL):
        os.remove(OUT_JSONL)
    if os.path.exists(OUT_CSV):
        os.remove(OUT_CSV)

    use_auth = use_praw()
    print("Mode:", "PRAW (auth)" if use_auth else "Fallback (no-auth)")
    records = crawl_with_praw() if use_auth else crawl_with_fallback()

    # Basic terminal dump of what we got (if PRAW path didn’t print along the way)
    for i, r in enumerate(records, 1):
        terminal_print(i, r["title"], r["url"], r["date"], r["text"])

    save_jsonl(records, OUT_JSONL)
    save_csv(records, OUT_CSV)
    print(f"\nSaved {len(records)} posts to {OUT_JSONL}")
    print(f"Also saved CSV to {OUT_CSV}")

if __name__ == "__main__":
    main()

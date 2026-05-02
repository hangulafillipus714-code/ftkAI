# =============================================================================
#  Data Cleaning Engine v3 — Refactored Production Pipeline
# =============================================================================

import os
import re
import sys
import json
import math
import hashlib
import logging
import unicodedata
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set

# ── optional deps ─────────────────────────────────────────────────────────────
try:
    from langdetect import detect
    LANG_OK = True
except Exception:
    LANG_OK = False

try:
    from datasketch import MinHash, MinHashLSH
    MH_OK = True
except Exception:
    MH_OK = False


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class Config:
    allowed_languages: Set[str] = field(default_factory=lambda: {"en"})
    min_quality: float = 0.45
    min_words: int = 20
    dedup_threshold: float = 0.80
    minhash_perm: int = 128


# =============================================================================
# CLEANER
# =============================================================================

class Cleaner:
    _html = re.compile(r"<[^>]{1,60}>")
    _multi_nl = re.compile(r"\n{3,}")
    _ws = re.compile(r"[ \t]{2,}")

    def normalize(self, t: str) -> str:
        t = unicodedata.normalize("NFC", t)
        return (
            t.replace("\u2018", "'")
             .replace("\u2019", "'")
             .replace("\u201c", '"')
             .replace("\u201d", '"')
             .replace("\u2013", "-")
             .replace("\u2014", "--")
             .replace("\u2026", "...")
             .replace("\ufeff", "")
        )

    def clean(self, t: str) -> str:
        t = self._html.sub(" ", t)
        t = self._ws.sub(" ", t)
        t = self._multi_nl.sub("\n\n", t)
        return t.strip()


# =============================================================================
# QUALITY SCORER
# =============================================================================

class Scorer:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def score(self, text: str) -> float:
        words = text.split()
        if len(words) < self.cfg.min_words:
            return 0.0

        uniq = len(set(words)) / len(words)
        vocab = min(uniq / 0.6, 1.0)

        avg_len = sum(len(w) for w in words) / len(words)
        length = 1 - min(abs(avg_len - 5) / 5, 1)

        digits = sum(c.isdigit() for c in text) / max(len(text), 1)
        digit = max(1 - digits * 5, 0)

        caps = sum(c.isupper() for c in text if c.isalpha()) / max(1, len([c for c in text if c.isalpha()]))
        caps = max(1 - caps * 2, 0)

        punct = sum(not c.isalnum() and not c.isspace() for c in text) / max(len(text), 1)
        punct = 1 if 0.03 <= punct <= 0.15 else max(0, 1 - punct)

        return round((vocab + length + digit + caps + punct) / 5, 4)


# =============================================================================
# DEDUP
# =============================================================================

class Deduper:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.seen: set[str] = set()

        if MH_OK:
            self.lsh = MinHashLSH(threshold=cfg.dedup_threshold, num_perm=cfg.minhash_perm)

    def _sha(self, t: str) -> str:
        return hashlib.sha256(t.encode()).hexdigest()

    def is_dup(self, text: str) -> bool:
        norm = " ".join(text.lower().split())
        h = self._sha(norm)

        if h in self.seen:
            return True
        self.seen.add(h)
        return False


# =============================================================================
# PIPELINE
# =============================================================================

class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cleaner = Cleaner()
        self.scorer = Scorer(cfg)
        self.deduper = Deduper(cfg)

    def process_file(self, path: Path):
        text = path.read_text(errors="ignore")

        text = self.cleaner.normalize(text)
        text = self.cleaner.clean(text)

        paras = [p.strip() for p in text.split("\n\n") if p.strip()]

        kept = []
        stats = {"in": 0, "out": 0, "rejected": 0}

        for p in paras:
            stats["in"] += 1

            if self.deduper.is_dup(p):
                stats["rejected"] += 1
                continue

            score = self.scorer.score(p)
            if score < self.cfg.min_quality:
                stats["rejected"] += 1
                continue

            kept.append(p)
            stats["out"] += 1

        return "\n\n".join(kept), stats


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir")
    ap.add_argument("--output", default="cleaned")
    args = ap.parse_args()

    cfg = Config()
    pipe = Pipeline(cfg)

    inp = Path(args.input_dir)
    out = Path(args.output)
    out.mkdir(exist_ok=True)

    global_stats = {"files": 0, "in": 0, "out": 0, "rejected": 0}

    for f in inp.glob("*.txt"):
        cleaned, st = pipe.process_file(f)

        global_stats["files"] += 1
        global_stats["in"] += st["in"]
        global_stats["out"] += st["out"]
        global_stats["rejected"] += st["rejected"]

        (out / f.name).write_text(cleaned, encoding="utf-8")

    print(json.dumps(global_stats, indent=2))


if __name__ == "__main__":
    main()
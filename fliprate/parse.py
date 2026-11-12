"""Parse 'ANSWER: X' / 'CONFIDENCE: NN' from model output."""
from __future__ import annotations
import re

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_ANS = re.compile(r"ANSWER\s*[:=\-]?\s*\(?\**\s*([ABCD])\b", re.I)
_ANS2 = re.compile(r"\banswer\s+is\s*:?\s*\(?\**([ABCD])\b", re.I)
_CONF = re.compile(r"CONFIDENCE\s*[:=\-]?\s*\**\s*(\d{1,3})(?:\.\d+)?\s*%?", re.I)


def parse_reply(text: str) -> tuple[str | None, int | None]:
    """Return (letter or None, confidence 0-100 or None). Uses the LAST labelled match."""
    t = _THINK.sub("", text or "")
    t = t.replace("*", "")
    a = _ANS.findall(t) or _ANS2.findall(t)
    c = _CONF.findall(t)
    letter = a[-1].upper() if a else None
    conf = None
    if c:
        v = int(c[-1])
        if 0 <= v <= 100:
            conf = v
    return letter, conf

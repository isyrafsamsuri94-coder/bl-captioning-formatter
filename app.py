import re
from dataclasses import dataclass
from typing import List, Set, Optional, Tuple

import streamlit as st
import spacy
from spacy.matcher import PhraseMatcher, Matcher

# ============================================================
# spaCy model loading (cached for Streamlit Cloud)
# ============================================================

@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

nlp = load_nlp()

# ============================================================
# Configuration
# ============================================================

@dataclass
class SubtitleCfg:
    max_len: int = 60
    max_lines: int = 2
    block_budget: int = 120
    preserve_paragraph_breaks: bool = True

# ============================================================
# Normalization
# ============================================================

def normalize_ws(text: str) -> str:
    """Normalize whitespace only; preserve punctuation and words."""
    return re.sub(r"[ \t]+", " ", text).strip()

def normalize_for_compare(text: str) -> str:
    """Used only for validation comparisons."""
    return re.sub(r"\s+", " ", text).strip()

# ============================================================
# Phrase protection configuration
# ============================================================

NEVER_BREAK_PHRASES = [
    "as well as",
    "in order to",
    "due to",
    "because of",
    "as a result",
    "in terms of",
    "according to",
    "rather than",
    "even though",
    "with respect to",
    "in relation to",
    "based on",
    "at least",
    "more than",
    "less than",
    "no more than",
    "as soon as possible",
    "at the end of the day",
    "in the meantime",
    "for the most part",
    "more or less",
    "goes through",
]

TITLES = {"mr", "mrs", "ms", "dr", "prof", "professor", "sir", "madam"}

UNITS = {
    "percent", "%", "km", "m", "cm", "mm", "kg", "g", "mg", "hz",
    "seconds", "second", "mins", "min", "minutes", "minute",
    "hours", "hour", "dollars", "dollar", "usd", "sgd", "eur", "gbp"
}

# ============================================================
# Protected spans
# ============================================================

def protected_spans(doc) -> List:
    spans = []

    # Phrase matcher
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(p) for p in NEVER_BREAK_PHRASES]
    matcher.add("NBP", patterns)

    for _, start, end in matcher(doc):
        spans.append(doc[start:end])

    # Named entities
    spans.extend(list(doc.ents))

    # Proper noun sequences
    i = 0
    while i < len(doc) - 1:
        if doc[i].pos_ == "PROPN" and doc[i+1].pos_ == "PROPN":
            start = i
            while i < len(doc) and doc[i].pos_ == "PROPN":
                i += 1
            spans.append(doc[start:i])
        i += 1

    return spans

def forbid_breaks_inside_spans(spans) -> Set[int]:
    forbid = set()
    for sp in spans:
        for tok in sp[:-1]:
            forbid.add(tok.i)
    return forbid

# ============================================================
# Break prohibition rules
# ============================================================

def forbidden_break_token_indices(doc) -> Set[int]:
    forbid: Set[int] = set()

    spans = protected_spans(doc)
    forbid |= forbid_breaks_inside_spans(spans)

    for i, tok in enumerate(doc[:-1]):
        # Compound nouns
        if tok.dep_ == "compound":
            forbid.add(i)

        # Verb cohesion
        if tok.dep_ in {"aux", "neg", "prt"}:
            forbid.add(i)

        # Prepositions
        if tok.pos_ == "ADP":
            forbid.add(i)

        # Titles and units
        if tok.lower_ in TITLES or tok.lower_ in UNITS:
            forbid.add(i)

    return forbid

# ============================================================
# Helpers
# ============================================================

def is_break_after_function_word(line1: str) -> bool:
    return bool(
        re.search(
            r"\b(of|to|in|on|at|for|as|by|with|and|or|but)\s*$",
            line1.lower(),
        )
    )

def punctuation_bonus(line1: str) -> int:
    if re.search(r"[.!?]\s*$", line1):
        return 12
    if re.search(r"[;:]\s*$", line1):
        return 8
    if re.search(r",\s*$", line1):
        return 4
    return 0

# ============================================================
# Best 1–2 line split
# ============================================================

def best_two_line_split(block_text: str, cfg: SubtitleCfg) -> Optional[str]:
    text = normalize_ws(block_text)
    if len(text) <= cfg.max_len:
        return text

    doc = nlp(text)
    forbid = forbidden_break_token_indices(doc)

    best_score = -1
    best_split = None

    for i, tok in enumerate(doc[:-1]):
        if tok.i in forbid:
            continue

        split_pos = tok.idx + len(tok.text)
        left = text[:split_pos].rstrip()
        right = text[split_pos:].lstrip()

        if len(left) > cfg.max_len or len(right) > cfg.max_len:
            continue
        if is_break_after_function_word(left):
            continue

        score = punctuation_bonus(left)
        if score > best_score:
            best_score = score
            best_split = f"{left}\n{right}"

    return best_split

# ============================================================
# Transcript segmentation
# ============================================================

def split_into_paragraphs(text: str, preserve: bool) -> List[str]:
    text = text.strip()
    if not preserve:
        return [normalize_ws(text)]
    return [
        p.strip()
        for p in re.split(r"\n\s*\n", text)
        if p.strip()
    ]

def sentence_or_clause_units(paragraph: str) -> List[str]:
    para = normalize_ws(paragraph)
    if not para:
        return []

    doc = nlp(para)
    units = [sent.text.strip() for sent in doc.sents]

    final_units = []
    for u in units:
        if len(u) > 180:
            final_units.extend(re.split(r"[,;:]", u))
        else:
            final_units.append(u)

    return [normalize_ws(u) for u in final_units if u.strip()]

def pack_units_into_blocks(units: List[str], cfg: SubtitleCfg) -> List[str]:
    blocks = []
    buf = ""

    for u in units:
        candidate = f"{buf} {u}".strip() if buf else u
        if len(candidate) <= cfg.block_budget:
            buf = candidate
        else:
            if buf:
                blocks.append(buf)
            buf = u

    if buf:
        blocks.append(buf)

    return blocks

# ============================================================
# Fallback hard wrap
# ============================================================

def hard_wrap_to_blocks(text: str, cfg: SubtitleCfg) -> List[str]:
    text = normalize_ws(text)
    blocks = []

    while len(text) > cfg.block_budget:
        idx = text.rfind(" ", 0, cfg.block_budget)
        if idx == -1:
            idx = cfg.block_budget
        blocks.append(text[:idx].strip())
        text = text[idx:].strip()

    if text:
        blocks.append(text)

    return blocks

# ============================================================
# Remove trailing commas
# ============================================================

def remove_trailing_commas_from_blocks(blocks: List[str]) -> List[str]:
    out = []
    for b in blocks:
        if b == "":
            out.append(b)
            continue
        lines = b.split("\n")
        lines = [re.sub(r",\s*$", "", ln) for ln in lines]
        out.append("\n".join(lines))
    return out

# ============================================================
# Main formatter
# ============================================================

def format_transcript_rules(text: str, cfg: SubtitleCfg) -> List[str]:
    blocks_out = []

    paragraphs = split_into_paragraphs(text, cfg.preserve_paragraph_breaks)

    for p in paragraphs:
        units = sentence_or_clause_units(p)
        blocks = pack_units_into_blocks(units, cfg)

        for b in blocks:
            split = best_two_line_split(b, cfg)
            if split:
                blocks_out.append(split)
            else:
                hard_blocks = hard_wrap_to_blocks(b, cfg)
                blocks_out.extend(hard_blocks)

        blocks_out.append("")

    if blocks_out and blocks_out[-1] == "":
        blocks_out.pop()

    return blocks_out

def blocks_to_plain_text(blocks: List[str]) -> str:
    out_lines = []
    for b in blocks:
        if b == "":
            out_lines.append("")
        else:
            out_lines.append(b)
            out_lines.append("")
    return "\n".join(out_lines).rstrip()

def format_transcript_hybrid(text: str, cfg: SubtitleCfg) -> str:
    blocks = format_transcript_rules(text, cfg)
    blocks = remove_trailing_commas_from_blocks(blocks)
    return blocks_to_plain_text(blocks)

# ============================================================
# Streamlit UI
# ============================================================

st.title("Captioning Formatter Tool")
st.markdown(
    "Upload a text file. The formatted captions will be generated "
    "according to your rule-based subtitle constraints."
)

uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

if uploaded_file:
    input_text = uploaded_file.read().decode("utf-8")

    cfg = SubtitleCfg(
        max_len=60,
        max_lines=2,
        block_budget=120,
        preserve_paragraph_breaks=True,
    )

    with st.spinner("Formatting captions..."):
        output_text = format_transcript_hybrid(input_text, cfg)

    st.success("Formatting complete!")

    st.download_button(
        label="Download formatted text",
        data=output_text,
        file_name="formatted_captions.txt",
        mime="text/plain",
    )
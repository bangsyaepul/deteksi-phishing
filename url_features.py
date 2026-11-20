"""
url_features.py
================
Ekstraksi fitur dari TEKS URL saja (tanpa membuka halaman web).

Modul ini dipakai oleh DUA pihak yang sama persis:
  1. notebook  -> untuk membentuk data latih/uji set fitur URL
  2. app.py    -> untuk mengubah URL masukan pengguna menjadi fitur model

Dengan memakai satu modul yang sama, nilai fitur saat pelatihan dan saat
prediksi dijamin identik (tidak ada training-serving skew).

Fitur: 19 fitur asli PhiUSIIL yang bisa dihitung dari teks URL (URL_FEATURES).
Nama kolom sama persis dengan nama kolom pada dataset PhiUSIIL.
"""
import json
import math
import os
import re

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Daftar fitur (nama kolom URL_FEATURES sengaja sama dengan nama kolom PhiUSIIL,
# termasuk salah ejaan 'Degits' dan 'Spacial', agar mudah dibandingkan)
# --------------------------------------------------------------------------
URL_FEATURES = [
    "URLLength", "DomainLength", "IsDomainIP", "TLDLength", "NoOfSubDomain",
    "HasObfuscation", "NoOfObfuscatedChar", "ObfuscationRatio",
    "NoOfLettersInURL", "LetterRatioInURL", "NoOfDegitsInURL", "DegitRatioInURL",
    "NoOfEqualsInURL", "NoOfQMarkInURL", "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL", "SpacialCharRatioInURL",
    "IsHTTPS", "CharContinuationRate",
]

ALL_URL_BASED_FEATURES = list(URL_FEATURES)

# --------------------------------------------------------------------------
# Konfigurasi varian definisi. Dokumentasi resmi PhiUSIIL tidak memberi rumus
# eksak untuk beberapa fitur, sehingga notebook MENGKALIBRASI varian terbaik
# terhadap kolom asli dataset lalu menyimpannya ke url_features_config.json.
# --------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "subdomain": "labels_minus_2",   # labels_minus_2 | dots | dots_minus_1 | labels_minus_2_no_www
    "ccr": "runs_sum",               # runs_sum | pair_ratio | longest_run
    "special": "all_non_alnum",      # all_non_alnum | excl_scheme_dot
    "special_ratio": "other_only",   # other_only | all_special
    "obfuscation": "seq_count",      # seq_count | seq_count_x3
    "tld_ip": "last_label",          # last_label | zero
}
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "url_features_config.json")
CONFIG = dict(DEFAULT_CONFIG)


def load_config(path=CONFIG_FILE):
    if os.path.exists(path):
        try:
            with open(path) as f:
                CONFIG.update({k: v for k, v in json.load(f).items() if k in DEFAULT_CONFIG})
        except Exception:
            pass
    return CONFIG


def save_config(cfg, path=CONFIG_FILE):
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


load_config()

# --------------------------------------------------------------------------
# Parsing URL (manual, tidak melempar error pada URL yang aneh)
# --------------------------------------------------------------------------
_SCHEME_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.\-]*)://(.*)$", re.S)
_IPV4_RE = re.compile(r"^(\d{1,3})(\.\d{1,3}){3}$")
_OBF_RE = re.compile(r"%[0-9a-fA-F]{2}")
_LETTER_RE = re.compile(r"[A-Za-z]")
_DIGIT_RE = re.compile(r"[0-9]")


def normalize_url(url, assume_scheme="https://"):
    """Bersihkan spasi; tambahkan skema bila pengguna tidak mengetikkannya."""
    url = str(url).strip()
    if not _SCHEME_RE.match(url):
        url = assume_scheme + url.lstrip("/")
    return url


def split_url(url):
    """Pecah URL menjadi (scheme, host, path, query)."""
    url = str(url).strip()
    m = _SCHEME_RE.match(url)
    scheme, rest = (m.group(1).lower(), m.group(2)) if m else ("", url)
    cut = re.search(r"[/?#]", rest)
    authority = rest[:cut.start()] if cut else rest
    tail = rest[cut.start():] if cut else ""
    if "@" in authority:
        authority = authority.rsplit("@", 1)[1]
    if authority.startswith("["):                      # IPv6 literal
        end = authority.find("]")
        host = authority[:end + 1] if end != -1 else authority
    else:
        host = re.sub(r":\d*$", "", authority)
    q = tail.find("?")
    h = tail.find("#")
    path_end = min([i for i in (q, h) if i != -1], default=len(tail))
    path = tail[:path_end]
    query = ""
    if q != -1 and (h == -1 or q < h):
        query = tail[q + 1:(h if h != -1 else len(tail))]
    return scheme, host, path, query


def _char_type(ch):
    if ch.isascii() and ch.isalpha():
        return "L"
    if ch.isdigit():
        return "D"
    return "S"


def _char_continuation(url, variant):
    n = len(url)
    if n == 0:
        return 0.0
    types = [_char_type(c) for c in url]
    if variant == "pair_ratio":
        if n == 1:
            return 1.0
        return sum(types[i] == types[i + 1] for i in range(n - 1)) / (n - 1)
    longest = {"L": 0, "D": 0, "S": 0}
    run = 1
    for i in range(1, n + 1):
        if i < n and types[i] == types[i - 1]:
            run += 1
        else:
            longest[types[i - 1]] = max(longest[types[i - 1]], run)
            run = 1
    if variant == "longest_run":
        return max(longest.values()) / n
    return sum(longest.values()) / n          # runs_sum


def extract_features(url, cfg=None):
    """Hitung 19 fitur URL (nama = kolom PhiUSIIL) dari satu string URL."""
    cfg = cfg or CONFIG
    url = str(url).strip()
    scheme, host, path, query = split_url(url)
    n = max(len(url), 1)

    # ---- 19 fitur URL --------------------------------------------------
    is_ip = int(bool(_IPV4_RE.match(host)) or host.startswith("["))
    labels = [l for l in host.split(".") if l != ""]
    if is_ip and cfg["tld_ip"] == "zero":
        tld_len = 0
    else:
        tld_len = len(labels[-1]) if len(labels) > 1 or (labels and is_ip) else 0

    sub = cfg["subdomain"]
    if is_ip:
        n_sub = 0
    elif sub == "dots":
        n_sub = host.count(".")
    elif sub == "dots_minus_1":
        n_sub = max(host.count(".") - 1, 0)
    elif sub == "labels_minus_2_no_www":
        n_sub = max(len(labels) - (3 if host.lower().startswith("www.") else 2), 0)
    else:
        n_sub = max(len(labels) - 2, 0)

    n_obf = len(_OBF_RE.findall(url))
    if cfg["obfuscation"] == "seq_count_x3":
        n_obf *= 3
    n_letters = len(_LETTER_RE.findall(url))
    n_digits = len(_DIGIT_RE.findall(url))
    n_eq, n_q, n_amp = url.count("="), url.count("?"), url.count("&")
    n_other = len(url) - n_letters - n_digits - n_eq - n_q - n_amp
    if cfg["special"] == "excl_scheme_dot":
        n_other -= url.count(":") + url.count("/") + url.count(".")
    n_other = max(n_other, 0)
    special_total = n_other + (n_eq + n_q + n_amp if cfg["special_ratio"] == "all_special" else 0)

    f = {
        "URLLength": len(url),
        "DomainLength": len(host),
        "IsDomainIP": is_ip,
        "TLDLength": tld_len,
        "NoOfSubDomain": n_sub,
        "HasObfuscation": int(n_obf > 0),
        "NoOfObfuscatedChar": n_obf,
        "ObfuscationRatio": n_obf / n,
        "NoOfLettersInURL": n_letters,
        "LetterRatioInURL": n_letters / n,
        "NoOfDegitsInURL": n_digits,
        "DegitRatioInURL": n_digits / n,
        "NoOfEqualsInURL": n_eq,
        "NoOfQMarkInURL": n_q,
        "NoOfAmpersandInURL": n_amp,
        "NoOfOtherSpecialCharsInURL": n_other,
        "SpacialCharRatioInURL": special_total / n,
        "IsHTTPS": int(scheme == "https"),
        "CharContinuationRate": _char_continuation(url, cfg["ccr"]),
    }

    return f


def extract_dataframe(urls, cfg=None):
    """Ekstraksi fitur untuk banyak URL -> DataFrame (urutan kolom tetap)."""
    rows = [extract_features(u, cfg) for u in urls]
    return pd.DataFrame(rows, columns=ALL_URL_BASED_FEATURES)


# --------------------------------------------------------------------------
# Prediksi end-to-end dari teks URL mentah -> label (dipakai notebook & app)
# --------------------------------------------------------------------------
def load_bundle(path="phishing_model.joblib"):
    import joblib
    return joblib.load(path)


def predict_urls(urls, bundle, threshold=0.5, assume_scheme="https://", normalize=True):
    """
    urls   : list/Series teks URL mentah
    bundle : dict hasil joblib {pipeline, feature_names, ...}
    return : DataFrame [url, proba_phishing, prediksi]
    """
    urls = [str(u) for u in urls]
    used = [normalize_url(u, assume_scheme) for u in urls] if normalize else urls
    X = extract_dataframe(used)[bundle["feature_names"]]
    proba = bundle["pipeline"].predict_proba(X)[:, 1]
    return pd.DataFrame({
        "url": urls,
        "proba_phishing": proba,
        "prediksi": np.where(proba >= threshold, "PHISHING", "LEGITIMATE"),
    })

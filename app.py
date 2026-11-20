
import numpy as np
import pandas as pd
import streamlit as st
import url_features as uf

# ==========================================
# KONFIGURASI APLIKASI
# ==========================================

st.set_page_config(
    page_title="Detektor URL Phishing",
    page_icon="🛡️",
    layout="centered"
)


@st.cache_resource
def get_bundle():
    return uf.load_bundle("phishing_model.joblib")


@st.cache_data
def get_demo():
    return pd.read_csv("url_uji_dataset.csv")


bundle = get_bundle()
m = bundle["metrics"]

st.title("🛡️ Detektor URL Phishing")

st.caption(
    "Tidak Mencakup Pemblokiran URL maupun Deteksi Phishing secara realtime."
)

THRESHOLD = 0.50


# ==========================================
# FUNGSI MENAMPILKAN HASIL PREDIKSI
# ==========================================

def show_result(row):

    p = float(row["proba_phishing"])

    if row["prediksi"] == "PHISHING":

        st.error(
            f"⚠️ **PHISHING** — probabilitas {p:.1%}"
        )

    else:

        st.success(
            f"✅ **LEGITIMATE** — probabilitas phishing {p:.1%}"
        )

    st.progress(
        min(max(p, 0.0), 1.0)
    )


# ==========================================
# FUNGSI NORMALISASI URL
# ==========================================

def normalize_for_comparison(url):

    return str(
        uf.normalize_url(url)
    ).strip().lower()


# ==========================================
# FUNGSI PENCOCOKAN LABEL DATASET
# ==========================================

def check_dataset_label(url, prediction, demo):

    url_key = normalize_for_comparison(url)

    demo_keys = demo["url"].apply(
        normalize_for_comparison
    )

    data_cocok = demo[demo_keys == url_key]

    if not data_cocok.empty:

        label_asli = str(
            data_cocok.iloc[0]["label_dataset"]
        ).strip().upper()

        prediksi = str(
            prediction
        ).strip().upper()

        ok = prediksi == label_asli

        st.write(
            f"Label dataset: **{label_asli}** → "
            + (
                "✅ prediksi sesuai"
                if ok
                else "❌ prediksi berbeda"
            )
        )

    else:

        st.info(
            "URL tidak ditemukan dalam dataset uji PhiUSIIL. "
            "Label asli dataset tidak tersedia sehingga "
            "kesesuaian prediksi belum dapat dibandingkan."
        )


# ==========================================
# TAB APLIKASI
# ==========================================

tab1, tab2 = st.tabs([
    "Cek satu URL",
    "Uji dengan URL dataset PhiUSIIL"
])


# ==========================================
# TAB 1: CEK SATU URL
# ==========================================

with tab1:

    url = st.text_input(
        "Masukkan URL",
        placeholder="https://contoh-situs.com/login"
    )

    if st.button("Periksa", type="primary") and url.strip():

        # Prediksi URL
        res = uf.predict_urls(
            [url],
            bundle
        ).iloc[0]

        # Tampilkan hasil prediksi
        show_result(res)

        # Ekstraksi fitur URL
        feats = uf.extract_dataframe(
            [uf.normalize_url(url)]
        )[bundle["feature_names"]].T

        feats.columns = ["nilai"]

        with st.expander(
            "Lihat fitur yang diekstrak dari URL"
        ):

            st.dataframe(
                feats,
                use_container_width=True
            )

        # Pencocokan dengan label dataset
        demo = get_demo()

        check_dataset_label(
            url,
            res["prediksi"],
            demo
        )


# ==========================================
# TAB 2: UJI DENGAN URL DATASET PHIUSIIL
# ==========================================

with tab2:

    st.write(
        "Sampel URL dari **data uji** PhiUSIIL "
    )

    demo = get_demo()

    c1, c2 = st.columns(2)


    # --------------------------------------
    # TOMBOL AMBIL 1 URL ACAK
    # --------------------------------------

    if c1.button("🎲 Ambil 1 URL acak"):

        # Ambil satu URL secara acak
        r = demo.sample(1).iloc[0]

        # Prediksi URL
        res = uf.predict_urls(
            [r["url"]],
            bundle
        ).iloc[0]

        # Tampilkan URL
        st.code(
            r["url"],
            language=None
        )

        # Tampilkan hasil prediksi
        show_result(res)

        # Pencocokan label dataset
        check_dataset_label(
            r["url"],
            res["prediksi"],
            demo
        )


    # --------------------------------------
    # TOMBOL UJI 10 URL ACAK
    # --------------------------------------

    if c2.button("📊 Uji 10 URL acak"):

        # Ambil tepat 10 URL secara acak
        s = demo.sample(n=10)

        # Prediksi 10 URL
        out = uf.predict_urls(
            s["url"].tolist(),
            bundle
        )

        # Tambahkan label asli dataset
        out["label_dataset"] = (
            s["label_dataset"].values
        )

        # Bandingkan prediksi dengan label dataset
        out["sesuai"] = (
            out["prediksi"].astype(str).str.strip().str.upper()
            ==
            out["label_dataset"].astype(str).str.strip().str.upper()
        )

        # Hitung hasil evaluasi
        jumlah_sesuai = int(
            out["sesuai"].sum()
        )

        jumlah_tidak_sesuai = (
            len(out) - jumlah_sesuai
        )

        persentase = (
            out["sesuai"].mean() * 100
        )

        # Tampilkan metrik kesesuaian
        st.metric(
            "Kesesuaian dengan label dataset",
            f"{persentase:.2f}%",
            f"{jumlah_sesuai}/10 URL"
        )

        # Ringkasan hasil prediksi
        col1, col2 = st.columns(2)

        col1.metric(
            "Prediksi Sesuai",
            f"{jumlah_sesuai}/10"
        )

        col2.metric(
            "Prediksi Tidak Sesuai",
            f"{jumlah_tidak_sesuai}/10"
        )

        # Tampilkan seluruh hasil pengujian
        st.subheader(
            "Hasil Pengujian 10 URL Acak"
        )

        st.dataframe(
            out,
            use_container_width=True,
            hide_index=True
        )

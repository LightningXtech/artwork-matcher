import pandas as pd
import streamlit as st
import fitz
from PIL import Image
import io
import re
import numpy as np
from rapidfuzz import fuzz
from paddleocr import PaddleOCR

# -----------------------------
# STREAMLIT CONFIG
# -----------------------------
st.set_page_config(
    page_title="Artwork Matcher",
    layout="centered"
)

# -----------------------------
# LOAD OCR
# -----------------------------
@st.cache_resource
def load_ocr():
    return PaddleOCR(
        use_textline_orientation=True,
        lang='en'
    )

ocr = load_ocr()

# -----------------------------
# OCR
# -----------------------------
def ocr_page(page):

    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))

    img = Image.open(
        io.BytesIO(pix.tobytes("png"))
    ).convert("RGB")

    img_np = np.array(img)

    result = ocr.predict(img_np)

    text = ""

    try:
        for line in result:

            if "rec_texts" in line:
                text += " ".join(line["rec_texts"]) + " "

    except:
        pass

    return text


# -----------------------------
# TEXT EXTRACTION
# -----------------------------
def extract_text(file):

    file_bytes = file.read()

    doc = fitz.open(
        stream=file_bytes,
        filetype="pdf"
    )

    text = ""

    for page in doc:

        page_text = page.get_text()

        # OCR fallback
        if len(page_text.strip()) < 20:
            page_text += ocr_page(page)

        text += page_text + " "

    return text, len(doc)


# -----------------------------
# CLEAN TEXT
# -----------------------------
def clean(text):

    text = text.lower()

    text = re.sub(r'\d+', '', text)

    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# -----------------------------
# BASE ID
# -----------------------------
def extract_base_id(filename):

    return filename.split("-")[0]


# -----------------------------
# DIMENSION
# -----------------------------
def extract_dimension(text):

    text = text.lower()

    match = re.search(
        r'(\d+\s*mm\s*x\s*\d+\s*mm)',
        text
    )

    if match:
        return match.group(1)

    return "NA"


# -----------------------------
# LABEL
# -----------------------------
def extract_label(text):

    t = text.upper()

    if "CLADD" in t:
        return "CLADD"

    if "ADDL" in t:
        return "ADDL"

    if "ADDS" in t:
        return "ADDS"

    return "STD"


# -----------------------------
# MATERIAL BLOCKS
# -----------------------------
def extract_material(text):

    t = text.lower()

    mapping = {
        "body shell": "shell",
        "shell": "shell",
        "body lining": "lining",
        "lining": "lining",
        "padding": "padding",
        "interlayer": "interlayer",
        "main material": "main",
        "pockets": "pockets"
    }

    found = set()

    for k, v in mapping.items():

        if k in t:
            found.add(v)

    return sorted(list(found))


# -----------------------------
# GARMENT TYPE
# -----------------------------
def extract_garment(text):

    t = text.lower()

    if "coat" in t:
        return "coat"

    if "jacket" in t or "jkt" in t:
        return "jacket"

    if "windbreaker" in t:
        return "windbreaker"

    if "pant" in t:
        return "pants"

    if "tee" in t or "t m" in t:
        return "tshirt"

    return "unknown"


# -----------------------------
# STRUCTURE SIGNATURE
# -----------------------------
def structure_signature(blocks):

    return "|".join(sorted(blocks))


# -----------------------------
# LAYOUT
# -----------------------------
def extract_layout(text):

    t = text.lower()

    keys = [
        "rn#",
        "ca#",
        "made in",
        "importer",
        "warning"
    ]

    return sorted([
        k for k in keys if k in t
    ])


# -----------------------------
# INTERLAYER
# -----------------------------
def has_interlayer(text):

    return "interlayer" in text.lower()


# -----------------------------
# SECTIONS
# -----------------------------
def extract_sections(text):

    t = text.lower()

    material = clean(t)

    care = clean(t if "wash" in t else "")

    warning = clean(t if "warning" in t else "")

    return material, care, warning


# -----------------------------
# PROFILE
# -----------------------------
def build_profile(text, pages, filename):

    blocks = extract_material(text)

    mat, care, warn = extract_sections(text)

    return {
        "name": filename,
        "base_id": extract_base_id(filename),
        "pages": pages,
        "dimension": extract_dimension(text),
        "label": extract_label(text),
        "layout": extract_layout(text),
        "interlayer": has_interlayer(text),
        "structure": structure_signature(blocks),
        "garment": extract_garment(text),
        "mat_text": mat,
        "care_text": care,
        "warn_text": warn
    }


# -----------------------------
# HARD MATCH
# -----------------------------
def hard_match(a, b):

    return (
        a["pages"] == b["pages"] and
        a["dimension"] == b["dimension"] and
        a["label"] == b["label"] and
        a["interlayer"] == b["interlayer"] and
        a["garment"] == b["garment"]
    )


# -----------------------------
# LAYOUT MATCH
# -----------------------------
def layout_match(a, b):

    return len(
        set(a["layout"]) &
        set(b["layout"])
    ) >= 3


# -----------------------------
# SAFE MATCH
# -----------------------------
def is_match(a, b):

    # HARD CHECKS
    if not hard_match(a, b):
        return False

    # SAME BASE ID ONLY
    if a["base_id"] != b["base_id"]:
        return False

    # STRUCTURE CHECK
    if a["structure"] != b["structure"]:
        return False

    # LAYOUT CHECK
    if not layout_match(a, b):
        return False

    # TEXT SCORES
    mat_score = fuzz.ratio(
        a["mat_text"],
        b["mat_text"]
    )

    care_score = fuzz.ratio(
        a["care_text"],
        b["care_text"]
    )

    warn_score = fuzz.ratio(
        a["warn_text"],
        b["warn_text"]
    )

    return (
        mat_score >= 93 and
        care_score >= 90 and
        warn_score >= 88
    )


# -----------------------------
# REVIEW MATCH
# -----------------------------
def review_match(a, b):

    # ONLY DIFFERENT IDs
    if a["base_id"] == b["base_id"]:
        return False

    if not hard_match(a, b):
        return False

    if a["structure"] != b["structure"]:
        return False

    mat_score = fuzz.ratio(
        a["mat_text"],
        b["mat_text"]
    )

    care_score = fuzz.ratio(
        a["care_text"],
        b["care_text"]
    )

    warn_score = fuzz.ratio(
        a["warn_text"],
        b["warn_text"]
    )

    return (
        mat_score >= 96 and
        care_score >= 95 and
        warn_score >= 95
    )


# -----------------------------
# GROUP FILES
# -----------------------------
def group_files(profiles):

    groups = []

    used = set()

    for i in range(len(profiles)):

        if i in used:
            continue

        group = [i]

        for j in range(len(profiles)):

            if j == i or j in used:
                continue

            if all(
                is_match(profiles[k], profiles[j])
                for k in group
            ):
                group.append(j)

        for g in group:
            used.add(g)

        groups.append(group)

    return groups


# -----------------------------
# REVIEW PAIRS
# -----------------------------
def find_review_pairs(profiles):

    pairs = []

    for i in range(len(profiles)):

        for j in range(i + 1, len(profiles)):

            if review_match(
                profiles[i],
                profiles[j]
            ):
                pairs.append(
                    (profiles[i], profiles[j])
                )

    return pairs


# -----------------------------
# UI
# -----------------------------
st.title("🎯 Artwork Matcher (SMART PRODUCTION)")

# -----------------------------
# SESSION
# -----------------------------
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# -----------------------------
# UI
# -----------------------------
col1, col2 = st.columns([4, 1])

with col1:

    files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}"
    )

with col2:

    st.write("")
    st.write("")

    if st.button("Clear Uploads"):

        st.session_state.uploader_key += 1

        st.rerun()
# -----------------------------
# PROCESS
# -----------------------------
if files:

    profiles = []

    with st.spinner("Analyzing artwork..."):

        for f in files:

            text, pages = extract_text(f)

            profiles.append(
                build_profile(
                    text,
                    pages,
                    f.name
                )
            )

    # -----------------------------
    # MATCHING
    # -----------------------------
    groups = group_files(profiles)

    review_pairs = find_review_pairs(profiles)

    st.success("Matching complete")

    # -----------------------------
    # EXCEL EXPORT
    # -----------------------------
    excel_rows = []

    gid_excel = 1

    for g in groups:

        if len(g) < 2:
            continue

        for idx in g:

            excel_rows.append({
                "Group": f"Group {gid_excel}",
                "Filename": profiles[idx]["name"]
            })

        gid_excel += 1

    if excel_rows:

        df = pd.DataFrame(excel_rows)

        excel_buffer = io.BytesIO()

        with pd.ExcelWriter(
            excel_buffer,
            engine="openpyxl"
        ) as writer:

            df.to_excel(
                writer,
                index=False
            )

        st.download_button(
            label="📥 Download Grouped Results Excel",
            data=excel_buffer.getvalue(),
            file_name="grouped_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # -----------------------------
    # SAFE GROUPS
    # -----------------------------
    st.header("✅ Safe to Print")

    gid = 1

    has_safe = False

    for g in groups:

        if len(g) < 2:
            continue

        has_safe = True

        st.subheader(f"Group {gid}")

        gid += 1

        for idx in g:

            st.write(
                f"- {profiles[idx]['name']}"
            )

    if not has_safe:

        st.info("No safe printable groups found.")

    # -----------------------------
    # REVIEW SECTION
    # -----------------------------
    if review_pairs:

        st.header("⚠️ Possible Matches (Review)")

        for a, b in review_pairs:

            st.write(
                f"{a['name']}  ↔  {b['name']}"
            )
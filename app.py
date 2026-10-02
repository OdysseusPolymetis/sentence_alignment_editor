import hashlib
import html
import re
import time

import pandas as pd
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer


st.set_page_config(
    page_title="LaBSE Aligner",
    page_icon="🔗",
    layout="wide",
)

st.title("🔗 LaBSE Aligner")
st.caption(
    "Alignement exploratoire de deux textes par embeddings, similarité cosinus "
    "et validation humaine."
)


# ----------------------------
# Utilitaires
# ----------------------------

def decode_uploaded_file(uploaded_file):
    if uploaded_file is None:
        return ""
    raw = uploaded_file.getvalue()
    for encoding in ("utf-8-sig", "utf-8", "utf-16", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def segment_text(text, mode):
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    if not text:
        return []

    if mode == "Lignes":
        segments = [line.strip() for line in text.split("\n")]

    elif mode == "Paragraphes":
        segments = [
            re.sub(r"\s+", " ", p).strip()
            for p in re.split(r"\n\s*\n+", text)
        ]

    else:  # Phrases — segmentation simple pour la V1
        cleaned = re.sub(r"\s+", " ", text).strip()
        segments = re.split(r"(?<=[.!?;··])\s+", cleaned)

    return [s for s in segments if s]


def make_signature(text_a, text_b, model_name, segmentation):
    payload = "\n---A---\n".join(
        [model_name, segmentation, text_a, text_b]
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@st.cache_resource(show_spinner=False, max_entries=1)
def load_model(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        model = SentenceTransformer(
            model_name,
            device=device,
            model_kwargs={"torch_dtype": torch.float16},
        )
    else:
        model = SentenceTransformer(model_name, device=device)

    return model


def encode_with_fallback(model, segments, requested_batch_size):
    """
    Encode en GPU. Si CUDA manque de VRAM, divise automatiquement
    le batch par deux jusqu'à réussite.
    """
    batch_size = int(requested_batch_size)

    while batch_size >= 1:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            embeddings = model.encode(
                segments,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_tensor=True,
                show_progress_bar=False,
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            return embeddings, batch_size

        except torch.OutOfMemoryError:
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            torch.cuda.empty_cache()


def cosine_for_text_pair(model, text_a, text_b):
    """Recalcule le cosine entre deux segments édités."""
    embeddings = model.encode(
        [text_a, text_b],
        batch_size=2,
        normalize_embeddings=True,
        convert_to_tensor=True,
        show_progress_bar=False,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return float(
        torch.dot(
            embeddings[0].float(),
            embeddings[1].float(),
        ).item()
    )


def build_matches(
    emb_a,
    emb_b,
    segments_a,
    segments_b,
    threshold,
    top_k,
    show_all,
    chunk_size=512,
):
    """
    Recherche A → B par blocs pour éviter de matérialiser toute la
    matrice de similarités en mémoire. Le produit scalaire est calculé
    en FP32 sur des embeddings normalisés, donc équivaut au cosinus.
    """
    rows = []
    b32 = emb_b.float()
    n_b = len(segments_b)

    if n_b == 0:
        return pd.DataFrame()

    k = min(int(top_k), n_b)

    for start in range(0, len(segments_a), chunk_size):
        end = min(start + chunk_size, len(segments_a))
        scores = emb_a[start:end].float() @ b32.T

        if show_all:
            local_i, j = torch.where(scores >= threshold)
            values = scores[local_i, j]

            for li, jj, score in zip(
                local_i.tolist(),
                j.tolist(),
                values.tolist(),
            ):
                i = start + li
                rows.append(
                    {
                        "pair_id": f"{i}::{jj}",
                        "id_a": i + 1,
                        "texte_a": segments_a[i],
                        "id_b": jj + 1,
                        "texte_b": segments_b[jj],
                        "cosine": float(score),
                    }
                )

        else:
            values, indices = torch.topk(scores, k=k, dim=1)

            mask = values >= threshold
            local_i, rank = torch.where(mask)

            for li, rr in zip(local_i.tolist(), rank.tolist()):
                i = start + li
                jj = int(indices[li, rr].item())
                score = float(values[li, rr].item())

                rows.append(
                    {
                        "pair_id": f"{i}::{jj}",
                        "id_a": i + 1,
                        "texte_a": segments_a[i],
                        "id_b": jj + 1,
                        "texte_b": segments_b[jj],
                        "cosine": score,
                    }
                )

        del scores

    if not rows:
        return pd.DataFrame(
            columns=[
                "pair_id", "id_a", "texte_a",
                "id_b", "texte_b", "cosine"
            ]
        )

    df = pd.DataFrame(rows)
    return df.sort_values("cosine", ascending=False).reset_index(drop=True)


# ----------------------------
# État de session
# ----------------------------

if "annotations" not in st.session_state:
    st.session_state.annotations = {}

if "alignment" not in st.session_state:
    st.session_state.alignment = None

if "segment_edits" not in st.session_state:
    st.session_state.segment_edits = {}


# ----------------------------
# Barre latérale
# ----------------------------

with st.sidebar:
    st.header("Réglages")

    model_name = st.text_input(
        "Modèle Sentence Transformer",
        value="sentence-transformers/LaBSE",
        help=(
            "Nom Hugging Face ou chemin vers un modèle Sentence Transformer local."
        ),
    )

    segmentation = st.selectbox(
        "Segmentation",
        ["Phrases", "Lignes", "Paragraphes"],
        index=0,
    )

    batch_size = st.number_input(
        "Batch size",
        min_value=1,
        max_value=1024,
        value=128,
        step=16,
        help="En cas d'erreur mémoire CUDA, l'application réduit automatiquement le batch.",
    )

    st.divider()

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        st.success(f"GPU : {gpu_name}")
        st.caption(f"CUDA actif · {total_vram:.2f} Gio de VRAM")
    else:
        st.warning("CUDA indisponible : calcul sur CPU.")


# ----------------------------
# Entrées
# ----------------------------

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Texte A")
    upload_a = st.file_uploader(
        "Charger un fichier A (.txt)",
        type=["txt"],
        key="upload_a",
    )
    pasted_a = st.text_area(
        "Ou coller le texte A",
        height=280,
        key="pasted_a",
    )

with col_b:
    st.subheader("Texte B")
    upload_b = st.file_uploader(
        "Charger un fichier B (.txt)",
        type=["txt"],
        key="upload_b",
    )
    pasted_b = st.text_area(
        "Ou coller le texte B",
        height=280,
        key="pasted_b",
    )

text_a = decode_uploaded_file(upload_a) if upload_a is not None else pasted_a
text_b = decode_uploaded_file(upload_b) if upload_b is not None else pasted_b

segments_a = segment_text(text_a, segmentation)
segments_b = segment_text(text_b, segmentation)

m1, m2, m3 = st.columns(3)
m1.metric("Segments A", len(segments_a))
m2.metric("Segments B", len(segments_b))
m3.metric("Comparaisons théoriques", f"{len(segments_a) * len(segments_b):,}".replace(",", " "))

current_signature = make_signature(
    text_a,
    text_b,
    model_name.strip(),
    segmentation,
)


# ----------------------------
# Embeddings
# ----------------------------

if st.button(
    "Calculer les embeddings",
    type="primary",
    use_container_width=True,
):
    if not text_a.strip() or not text_b.strip():
        st.error("Il faut fournir les deux textes.")
        st.stop()

    if not model_name.strip():
        st.error("Il faut indiquer un modèle.")
        st.stop()

    try:
        with st.spinner(f"Chargement de {model_name}…"):
            model = load_model(model_name.strip())

        start = time.perf_counter()

        with st.spinner(f"Embedding du texte A ({len(segments_a)} segments)…"):
            emb_a, actual_batch_a = encode_with_fallback(
                model,
                segments_a,
                batch_size,
            )

        with st.spinner(f"Embedding du texte B ({len(segments_b)} segments)…"):
            emb_b, actual_batch_b = encode_with_fallback(
                model,
                segments_b,
                batch_size,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

        peak_vram = None
        if torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated() / 1024**3

        st.session_state.alignment = {
            "signature": current_signature,
            "model_name": model_name.strip(),
            "segments_a": segments_a,
            "segments_b": segments_b,
            "emb_a": emb_a,
            "emb_b": emb_b,
            "batch_a": actual_batch_a,
            "batch_b": actual_batch_b,
            "elapsed": elapsed,
            "peak_vram": peak_vram,
        }

    except Exception as exc:
        st.exception(exc)
        st.stop()


# ----------------------------
# Résultats
# ----------------------------

alignment = st.session_state.alignment

if alignment is not None:
    if alignment["signature"] != current_signature:
        st.warning(
            "Les textes, le modèle ou la segmentation ont changé depuis le dernier calcul. "
            "Clique sur « Calculer les embeddings » pour mettre les résultats à jour."
        )
        st.stop()

    st.success(
        f"Embeddings calculés en {alignment['elapsed']:.2f} s "
        f"avec batch A={alignment['batch_a']} et B={alignment['batch_b']}."
    )

    if alignment["peak_vram"] is not None:
        st.caption(
            f"Pic mémoire PyTorch observé : {alignment['peak_vram']:.2f} Gio."
        )

    st.divider()
    st.subheader("Correspondances")

    f1, f2, f3 = st.columns([2, 1, 2])

    with f1:
        threshold = st.slider(
            "Seuil de similarité cosinus",
            min_value=0.0,
            max_value=1.0,
            value=0.70,
            step=0.01,
        )

    with f2:
        top_k = st.number_input(
            "Top-k par segment A",
            min_value=1,
            max_value=100,
            value=5,
            step=1,
        )

    with f3:
        show_all = st.toggle(
            "Afficher toutes les paires au-dessus du seuil",
            value=False,
            help="Si activé, le top-k est ignoré.",
        )

    with st.spinner("Calcul des similarités…"):
        matches = build_matches(
            alignment["emb_a"],
            alignment["emb_b"],
            alignment["segments_a"],
            alignment["segments_b"],
            threshold=float(threshold),
            top_k=int(top_k),
            show_all=show_all,
        )

    st.write(f"**{len(matches):,} correspondance(s)**".replace(",", " "))

    if matches.empty:
        st.info("Aucune correspondance ne dépasse ce seuil.")
    else:
        if len(matches) > 100_000:
            st.warning(
                "Le résultat contient plus de 100 000 correspondances. "
                "La relecture reste possible, mais la vue d'ensemble peut devenir lourde."
            )

        review_identity = hashlib.md5(
            (
                alignment["signature"]
                + f"|{threshold:.3f}|{top_k}|{show_all}"
            ).encode("utf-8")
        ).hexdigest()[:12]

        # ------------------------------------------------------------
        # Relecture détaillée + correction locale de segmentation
        # ------------------------------------------------------------
        st.subheader("Relecture détaillée")

        st.markdown(
            """
            <style>
            .context-box {
                padding: 0.65rem 0.8rem;
                border: 1px solid rgba(128, 128, 128, 0.22);
                border-radius: 0.45rem;
                background: rgba(128, 128, 128, 0.035);
                white-space: pre-wrap;
                overflow-wrap: anywhere;
                line-height: 1.48;
                margin-bottom: 0.4rem;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        position_key = f"review_position_{review_identity}"
        if position_key not in st.session_state:
            st.session_state[position_key] = 1

        st.session_state[position_key] = min(
            max(int(st.session_state[position_key]), 1),
            len(matches),
        )

        def move_review(delta):
            st.session_state[position_key] = min(
                max(st.session_state[position_key] + delta, 1),
                len(matches),
            )

        nav_prev, nav_pos, nav_next = st.columns([1, 2, 1])

        with nav_prev:
            st.button(
                "← Précédent",
                key=f"prev_{review_identity}",
                on_click=move_review,
                args=(-1,),
                disabled=st.session_state[position_key] <= 1,
                use_container_width=True,
            )

        with nav_pos:
            st.number_input(
                "Correspondance à relire",
                min_value=1,
                max_value=len(matches),
                step=1,
                key=position_key,
            )

        with nav_next:
            st.button(
                "Suivant →",
                key=f"next_{review_identity}",
                on_click=move_review,
                args=(1,),
                disabled=st.session_state[position_key] >= len(matches),
                use_container_width=True,
            )

        current_index = int(st.session_state[position_key]) - 1
        current_row = matches.iloc[current_index]
        pair_id = current_row["pair_id"]

        # On scope les éditions au corpus/modèle courant pour éviter qu'une
        # paire 12::45 d'un autre corpus récupère accidentellement les mêmes edits.
        edit_id = f"{alignment['signature']}::{pair_id}"

        original_a_index = int(current_row["id_a"]) - 1
        original_b_index = int(current_row["id_b"]) - 1

        original_text_a = str(current_row["texte_a"])
        original_text_b = str(current_row["texte_b"])

        if edit_id not in st.session_state.segment_edits:
            st.session_state.segment_edits[edit_id] = {
                "texte_a": original_text_a,
                "texte_b": original_text_b,
                "a_start": original_a_index,
                "a_end": original_a_index,
                "b_start": original_b_index,
                "b_end": original_b_index,
                "cosine_edite": None,
                "score_text_a": None,
                "score_text_b": None,
            }

        edit_state = st.session_state.segment_edits[edit_id]

        edit_a_key = f"edit_a_{hashlib.md5(edit_id.encode()).hexdigest()[:16]}"
        edit_b_key = f"edit_b_{hashlib.md5(edit_id.encode()).hexdigest()[:16]}"

        if edit_a_key not in st.session_state:
            st.session_state[edit_a_key] = edit_state["texte_a"]
        if edit_b_key not in st.session_state:
            st.session_state[edit_b_key] = edit_state["texte_b"]

        def add_neighbor(side, direction):
            edit = st.session_state.segment_edits[edit_id]

            if side == "a":
                segments = alignment["segments_a"]
                text_key = edit_a_key
                start_key, end_key = "a_start", "a_end"
            else:
                segments = alignment["segments_b"]
                text_key = edit_b_key
                start_key, end_key = "b_start", "b_end"

            if direction == "prev":
                neighbor_index = edit[start_key] - 1
                if neighbor_index >= 0:
                    neighbor = segments[neighbor_index]
                    st.session_state[text_key] = (
                        neighbor + " " + st.session_state[text_key]
                    ).strip()
                    edit[start_key] = neighbor_index

            elif direction == "next":
                neighbor_index = edit[end_key] + 1
                if neighbor_index < len(segments):
                    neighbor = segments[neighbor_index]
                    st.session_state[text_key] = (
                        st.session_state[text_key] + " " + neighbor
                    ).strip()
                    edit[end_key] = neighbor_index

        def reset_side(side):
            edit = st.session_state.segment_edits[edit_id]

            if side == "a":
                st.session_state[edit_a_key] = original_text_a
                edit["texte_a"] = original_text_a
                edit["a_start"] = original_a_index
                edit["a_end"] = original_a_index
            else:
                st.session_state[edit_b_key] = original_text_b
                edit["texte_b"] = original_text_b
                edit["b_start"] = original_b_index
                edit["b_end"] = original_b_index

        st.caption(
            f"Correspondance {current_index + 1} / {len(matches)} · "
            f"cosine original = {current_row['cosine']:.4f}"
        )

        st.info(
            "Tu peux modifier directement les deux segments. "
            "Pour rallonger un alignement, ajoute le segment précédent ou suivant ; "
            "pour le raccourcir, édite simplement le texte dans la zone."
        )

        edit_col_a, edit_col_b = st.columns(2)

        # ----------------------------
        # Côté A
        # ----------------------------
        with edit_col_a:
            st.markdown(
                f"#### Texte A · segment {int(current_row['id_a'])}"
            )

            a_prev = (
                alignment["segments_a"][edit_state["a_start"] - 1]
                if edit_state["a_start"] > 0
                else None
            )
            a_next = (
                alignment["segments_a"][edit_state["a_end"] + 1]
                if edit_state["a_end"] + 1 < len(alignment["segments_a"])
                else None
            )

            with st.expander("Voir le contexte voisin A", expanded=False):
                if a_prev is not None:
                    st.caption(f"Segment précédent · {edit_state['a_start']}")
                    st.markdown(
                        f"<div class='context-box'>{html.escape(a_prev)}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption("Pas de segment précédent.")

                st.caption(
                    f"Plage actuellement incluse : "
                    f"{edit_state['a_start'] + 1}–{edit_state['a_end'] + 1}"
                )

                if a_next is not None:
                    st.caption(f"Segment suivant · {edit_state['a_end'] + 2}")
                    st.markdown(
                        f"<div class='context-box'>{html.escape(a_next)}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption("Pas de segment suivant.")

            a_b1, a_b2, a_b3 = st.columns(3)
            with a_b1:
                st.button(
                    "＋ précédent",
                    key=f"a_prev_{edit_a_key}",
                    on_click=add_neighbor,
                    args=("a", "prev"),
                    disabled=edit_state["a_start"] <= 0,
                    use_container_width=True,
                )
            with a_b2:
                st.button(
                    "＋ suivant",
                    key=f"a_next_{edit_a_key}",
                    on_click=add_neighbor,
                    args=("a", "next"),
                    disabled=edit_state["a_end"] + 1 >= len(alignment["segments_a"]),
                    use_container_width=True,
                )
            with a_b3:
                st.button(
                    "Réinitialiser",
                    key=f"a_reset_{edit_a_key}",
                    on_click=reset_side,
                    args=("a",),
                    use_container_width=True,
                )

            edited_text_a = st.text_area(
                "Segment A édité",
                height=260,
                key=edit_a_key,
                help=(
                    "Tu peux supprimer, ajouter ou déplacer du texte. "
                    "Le retour à la ligne est automatique."
                ),
            )

        # ----------------------------
        # Côté B
        # ----------------------------
        with edit_col_b:
            st.markdown(
                f"#### Texte B · segment {int(current_row['id_b'])}"
            )

            b_prev = (
                alignment["segments_b"][edit_state["b_start"] - 1]
                if edit_state["b_start"] > 0
                else None
            )
            b_next = (
                alignment["segments_b"][edit_state["b_end"] + 1]
                if edit_state["b_end"] + 1 < len(alignment["segments_b"])
                else None
            )

            with st.expander("Voir le contexte voisin B", expanded=False):
                if b_prev is not None:
                    st.caption(f"Segment précédent · {edit_state['b_start']}")
                    st.markdown(
                        f"<div class='context-box'>{html.escape(b_prev)}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption("Pas de segment précédent.")

                st.caption(
                    f"Plage actuellement incluse : "
                    f"{edit_state['b_start'] + 1}–{edit_state['b_end'] + 1}"
                )

                if b_next is not None:
                    st.caption(f"Segment suivant · {edit_state['b_end'] + 2}")
                    st.markdown(
                        f"<div class='context-box'>{html.escape(b_next)}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption("Pas de segment suivant.")

            b_b1, b_b2, b_b3 = st.columns(3)
            with b_b1:
                st.button(
                    "＋ précédent",
                    key=f"b_prev_{edit_b_key}",
                    on_click=add_neighbor,
                    args=("b", "prev"),
                    disabled=edit_state["b_start"] <= 0,
                    use_container_width=True,
                )
            with b_b2:
                st.button(
                    "＋ suivant",
                    key=f"b_next_{edit_b_key}",
                    on_click=add_neighbor,
                    args=("b", "next"),
                    disabled=edit_state["b_end"] + 1 >= len(alignment["segments_b"]),
                    use_container_width=True,
                )
            with b_b3:
                st.button(
                    "Réinitialiser",
                    key=f"b_reset_{edit_b_key}",
                    on_click=reset_side,
                    args=("b",),
                    use_container_width=True,
                )

            edited_text_b = st.text_area(
                "Segment B édité",
                height=260,
                key=edit_b_key,
                help=(
                    "Tu peux supprimer, ajouter ou déplacer du texte. "
                    "Le retour à la ligne est automatique."
                ),
            )

        # Sauvegarde immédiate du texte édité.
        edit_state["texte_a"] = edited_text_a
        edit_state["texte_b"] = edited_text_b

        st.markdown("##### Similarité après correction")

        score_is_current = (
            edit_state["cosine_edite"] is not None
            and edit_state["score_text_a"] == edited_text_a
            and edit_state["score_text_b"] == edited_text_b
        )

        score_c1, score_c2, score_c3 = st.columns([1, 1, 2])

        with score_c1:
            st.metric(
                "Cosine original",
                f"{float(current_row['cosine']):.4f}",
            )

        with score_c2:
            if score_is_current:
                st.metric(
                    "Cosine édité",
                    f"{edit_state['cosine_edite']:.4f}",
                    delta=(
                        f"{edit_state['cosine_edite'] - float(current_row['cosine']):+.4f}"
                    ),
                )
            else:
                st.metric("Cosine édité", "—")

        with score_c3:
            if st.button(
                "Recalculer le cosine des segments édités",
                key=f"recalc_{edit_a_key}",
                type="secondary",
                use_container_width=True,
            ):
                if not edited_text_a.strip() or not edited_text_b.strip():
                    st.error("Les deux segments édités doivent contenir du texte.")
                else:
                    with st.spinner("Recalcul du score…"):
                        model = load_model(alignment["model_name"])
                        new_score = cosine_for_text_pair(
                            model,
                            edited_text_a,
                            edited_text_b,
                        )

                    edit_state["cosine_edite"] = new_score
                    edit_state["score_text_a"] = edited_text_a
                    edit_state["score_text_b"] = edited_text_b
                    st.rerun()

        if not score_is_current and (
            edited_text_a != original_text_a
            or edited_text_b != original_text_b
        ):
            st.caption(
                "Les segments ont été modifiés : le cosine édité doit être recalculé."
            )

        # ------------------------------------------------------------
        # Annotation humaine
        # ------------------------------------------------------------
        annotation = st.session_state.annotations.get(
            edit_id,
            {"decision": "À revoir", "certitude": 50},
        )

        decision_key = f"decision_{hashlib.md5(edit_id.encode()).hexdigest()[:16]}"
        confidence_key = f"confidence_{hashlib.md5(edit_id.encode()).hexdigest()[:16]}"

        if decision_key not in st.session_state:
            st.session_state[decision_key] = annotation["decision"]

        if confidence_key not in st.session_state:
            st.session_state[confidence_key] = int(annotation["certitude"])

        annot_col_1, annot_col_2 = st.columns([2, 3])

        with annot_col_1:
            decision = st.radio(
                "Intertexte ?",
                ["Oui", "Non", "À revoir"],
                horizontal=True,
                key=decision_key,
            )

        with annot_col_2:
            confidence = st.slider(
                "Certitude (%)",
                min_value=0,
                max_value=100,
                step=1,
                key=confidence_key,
            )

        st.session_state.annotations[edit_id] = {
            "decision": decision,
            "certitude": int(confidence),
        }

        # ------------------------------------------------------------
        # Réinjecte annotations + éditions dans la vue globale et l'export
        # ------------------------------------------------------------
        decisions = []
        confidences = []
        edited_a_values = []
        edited_b_values = []
        edited_cosines = []
        a_ranges = []
        b_ranges = []

        for _, match_row in matches.iterrows():
            pid = match_row["pair_id"]
            scoped_id = f"{alignment['signature']}::{pid}"

            saved_annotation = st.session_state.annotations.get(
                scoped_id,
                {"decision": "À revoir", "certitude": 50},
            )
            decisions.append(saved_annotation["decision"])
            confidences.append(saved_annotation["certitude"])

            saved_edit = st.session_state.segment_edits.get(scoped_id)

            if saved_edit is None:
                edited_a_values.append(str(match_row["texte_a"]))
                edited_b_values.append(str(match_row["texte_b"]))
                edited_cosines.append(None)
                a_ranges.append(str(int(match_row["id_a"])))
                b_ranges.append(str(int(match_row["id_b"])))
            else:
                edited_a_values.append(saved_edit["texte_a"])
                edited_b_values.append(saved_edit["texte_b"])

                score_valid = (
                    saved_edit["cosine_edite"] is not None
                    and saved_edit["score_text_a"] == saved_edit["texte_a"]
                    and saved_edit["score_text_b"] == saved_edit["texte_b"]
                )
                edited_cosines.append(
                    saved_edit["cosine_edite"] if score_valid else None
                )

                a_ranges.append(
                    f"{saved_edit['a_start'] + 1}-{saved_edit['a_end'] + 1}"
                    if saved_edit["a_start"] != saved_edit["a_end"]
                    else str(saved_edit["a_start"] + 1)
                )
                b_ranges.append(
                    f"{saved_edit['b_start'] + 1}-{saved_edit['b_end'] + 1}"
                    if saved_edit["b_start"] != saved_edit["b_end"]
                    else str(saved_edit["b_start"] + 1)
                )

        matches["decision"] = decisions
        matches["certitude"] = confidences
        matches["texte_a_edite"] = edited_a_values
        matches["texte_b_edite"] = edited_b_values
        matches["cosine_edite"] = edited_cosines
        matches["segments_a_utilises"] = a_ranges
        matches["segments_b_utilises"] = b_ranges

        with st.expander("Vue d'ensemble des correspondances", expanded=False):
            st.dataframe(
                matches[
                    [
                        "id_a",
                        "texte_a",
                        "id_b",
                        "texte_b",
                        "cosine",
                        "cosine_edite",
                        "decision",
                        "certitude",
                    ]
                ],
                hide_index=True,
                use_container_width=True,
                height=520,
                row_height=72,
                column_config={
                    "id_a": st.column_config.NumberColumn(
                        "ID A",
                        width="small",
                        format="%d",
                    ),
                    "texte_a": st.column_config.TextColumn(
                        "Texte A original",
                        width="large",
                    ),
                    "id_b": st.column_config.NumberColumn(
                        "ID B",
                        width="small",
                        format="%d",
                    ),
                    "texte_b": st.column_config.TextColumn(
                        "Texte B original",
                        width="large",
                    ),
                    "cosine": st.column_config.NumberColumn(
                        "Cosine original",
                        width="small",
                        format="%.4f",
                    ),
                    "cosine_edite": st.column_config.NumberColumn(
                        "Cosine édité",
                        width="small",
                        format="%.4f",
                    ),
                    "decision": st.column_config.TextColumn(
                        "Intertexte ?",
                        width="small",
                    ),
                    "certitude": st.column_config.NumberColumn(
                        "Certitude (%)",
                        width="small",
                        format="%d",
                    ),
                },
            )

        # Export : on garde explicitement original + version d'expertise.
        export_df = pd.DataFrame(
            {
                "model": alignment["model_name"],
                "threshold": float(threshold),
                "top_k": "ALL" if show_all else int(top_k),
                "id_a": matches["id_a"],
                "segments_a_utilises": matches["segments_a_utilises"],
                "texte_a_original": matches["texte_a"],
                "texte_a_edite": matches["texte_a_edite"],
                "id_b": matches["id_b"],
                "segments_b_utilises": matches["segments_b_utilises"],
                "texte_b_original": matches["texte_b"],
                "texte_b_edite": matches["texte_b_edite"],
                "cosine_original": matches["cosine"],
                "cosine_edite": matches["cosine_edite"],
                "decision": matches["decision"],
                "certitude": matches["certitude"],
            }
        )

        csv_bytes = export_df.to_csv(
            index=False,
            encoding="utf-8-sig",
        ).encode("utf-8-sig")

        st.download_button(
            "Télécharger les alignements en CSV",
            data=csv_bytes,
            file_name="alignements_intertextes.csv",
            mime="text/csv",
            use_container_width=True,
        )
else:
    st.info(
        "Charge ou colle deux textes, puis clique sur « Calculer les embeddings »."
    )

"""
app.py
------
Streamlit UI for the Handwritten Digit Data Collection tool (Module 1).

Responsibilities of this file are limited to UI/orchestration only.
All image math lives in image_processing.py and all dataset I/O lives
in dataset_manager.py, per the modular architecture requirement.

Run with:
    streamlit run app.py
"""

import os
import hashlib

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from streamlit_autorefresh import st_autorefresh

from image_processing import process_canvas_image, ImageProcessingError
from dataset_manager import (
    save_sample,
    get_dataset_stats,
    DatasetValidationError,
    DATASET_FILENAME,
)

# --------------------------------------------------------------------------
# Page config & constants
# --------------------------------------------------------------------------
st.set_page_config(page_title="Handwritten Digit Data Collector", layout="wide")

DATASET_PATH = os.path.join(os.getcwd(), DATASET_FILENAME)
CANVAS_SIZE = 400

# How often (ms) we poll the canvas while Automatic mode is on, and how many
# consecutive unchanged polls count as "the user has paused drawing".
POLL_INTERVAL_MS = 1200
STABLE_POLLS_REQUIRED = 2


def _init_session_state():
    defaults = {
        "canvas_key": 0,          # bump to force-reset the canvas widget after a save
        "last_message": None,     # (type, text) tuple for feedback banners
        "last_hash": None,        # hash of the canvas content on the previous poll
        "stable_count": 0,        # how many polls in a row the hash hasn't changed
        "auto_saved_hash": None,  # hash already auto-saved, to avoid duplicate saves
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_session_state()


# --------------------------------------------------------------------------
# Sidebar: label selection + dataset analytics
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Sample Label")
    if "active_label" not in st.session_state:
        st.session_state.active_label = 0
    label = st.selectbox(
        "Select the digit you are drawing",
        options=list(range(10)),
        key="active_label",
    )

    st.divider()
    st.header("🤖 Automatic Mode")
    auto_mode = st.toggle(
        "Auto-save when I pause drawing",
        value=False,
        help=(
            "When on, the app watches your drawing and automatically "
            "processes + saves it about "
            f"{POLL_INTERVAL_MS * STABLE_POLLS_REQUIRED / 1000:.1f}s after you "
            "stop moving the pen — no button needed."
        ),
    )
    auto_advance = st.toggle(
        "Auto-advance digit label after each save (cycles 0→9)",
        value=True,
        disabled=not auto_mode,
    )

    st.divider()
    st.header("📊 Dataset Statistics")

    stats = get_dataset_stats(DATASET_PATH)
    st.metric("Total Samples", stats["total"])

    stats_df = pd.DataFrame(
        {
            "Digit": list(stats["per_digit"].keys()),
            "Count": list(stats["per_digit"].values()),
        }
    )
    st.bar_chart(stats_df.set_index("Digit"))
    st.dataframe(stats_df, hide_index=True, use_container_width=True)

    st.caption(f"Dataset file: `{DATASET_FILENAME}`")


# --------------------------------------------------------------------------
# Main layout
# --------------------------------------------------------------------------
st.title("✍️ Handwritten Digit Data Collection")
if auto_mode:
    st.write(
        "🤖 **Automatic mode is on.** Just draw — the sample saves itself "
        "shortly after you stop moving the pen, then the canvas clears for "
        "the next digit."
    )
    st_autorefresh(interval=POLL_INTERVAL_MS, key="auto_mode_poller")
else:
    st.write(
        "Draw a digit on the canvas below, then click **Save Sample** "
        "(or turn on Automatic mode in the sidebar to skip the button)."
    )

col_canvas, col_results = st.columns([1, 1])

# --------------------------------------------------------------------------
# Drawing canvas
# --------------------------------------------------------------------------
with col_canvas:
    st.subheader("Drawing Area")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=CANVAS_SIZE,
        width=CANVAS_SIZE,
        drawing_mode="freedraw",
        display_toolbar=True,  # gives Undo / Redo / Clear / Download for free
        key=f"canvas_{st.session_state.canvas_key}",
    )
    st.caption("Use the toolbar icons above the canvas to undo or clear your drawing.")

    save_clicked = st.button("💾 Save Sample", use_container_width=True, type="primary")


# --------------------------------------------------------------------------
# Live processing: recompute the preview automatically on every rerun,
# no "Process Image" button needed.
# --------------------------------------------------------------------------
result = None
if canvas_result.image_data is not None:
    raw_rgba = np.array(canvas_result.image_data)
    try:
        result = process_canvas_image(raw_rgba)
    except ImageProcessingError:
        result = None  # canvas is empty / nothing drawn yet -- not an error to show


def _canvas_hash(image_data) -> str:
    """Cheap content fingerprint used to detect when drawing has paused."""
    return hashlib.md5(np.asarray(image_data).tobytes()).hexdigest()


def _do_save(current_label) -> bool:
    """Validate + save the current sample. Returns True on success."""
    try:
        save_sample(DATASET_PATH, current_label, result["flattened"])
        st.session_state.last_message = (
            "success",
            f"Sample saved for digit '{current_label}'. Dataset updated.",
        )
        return True
    except DatasetValidationError as exc:
        st.session_state.last_message = ("error", f"Validation failed: {exc}")
        return False


# --------------------------------------------------------------------------
# Automatic mode: detect a drawing pause and auto-save without any click
# --------------------------------------------------------------------------
auto_saved_this_run = False
if auto_mode and canvas_result.image_data is not None and result is not None:
    current_hash = _canvas_hash(canvas_result.image_data)

    if current_hash == st.session_state.last_hash:
        st.session_state.stable_count += 1
    else:
        st.session_state.last_hash = current_hash
        st.session_state.stable_count = 0

    is_paused = st.session_state.stable_count >= STABLE_POLLS_REQUIRED
    already_saved = st.session_state.auto_saved_hash == current_hash

    if is_paused and not already_saved:
        if _do_save(label):
            st.session_state.auto_saved_hash = current_hash
            auto_saved_this_run = True
            if auto_advance:
                st.session_state.active_label = (label + 1) % 10
            st.session_state.canvas_key += 1  # clears canvas for the next sample
            st.session_state.last_hash = None
            st.session_state.stable_count = 0
            st.rerun()
elif canvas_result.image_data is None:
    # Canvas was cleared (e.g. via the toolbar trash icon) -- reset tracking
    st.session_state.last_hash = None
    st.session_state.stable_count = 0


# --------------------------------------------------------------------------
# Manual Save Sample handling (always available, even in automatic mode)
# --------------------------------------------------------------------------
if save_clicked and not auto_saved_this_run:
    if result is None:
        st.session_state.last_message = ("error", "Please draw a digit before saving.")
    else:
        if _do_save(label):
            st.session_state.canvas_key += 1  # clears canvas for the next sample
            st.session_state.last_hash = None
            st.session_state.stable_count = 0
            st.rerun()


# --------------------------------------------------------------------------
# Feedback banner
# --------------------------------------------------------------------------
if st.session_state.last_message is not None:
    msg_type, msg_text = st.session_state.last_message
    if msg_type == "success":
        st.success(msg_text)
    else:
        st.error(msg_text)


# --------------------------------------------------------------------------
# Results display
# --------------------------------------------------------------------------
with col_results:
    st.subheader("Processing Results")

    if result is None:
        st.info("Start drawing a digit — a live preview will appear here.")
    else:
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.caption("Original Image")
            st.image(result["original_gray"], clamp=True, use_container_width=True)
        with img_col2:
            st.caption("Processed Image (28x28)")
            # Scale up for visibility only; underlying data stays 28x28
            st.image(result["processed_28x28"], clamp=True, width=200)

        processed = result["processed_28x28"]
        st.markdown("**Image Summary**")
        summary_cols = st.columns(4)
        summary_cols[0].metric("Width", processed.shape[1])
        summary_cols[1].metric("Height", processed.shape[0])
        summary_cols[2].metric("Total Pixels", processed.size)
        summary_cols[3].metric("Pixel Range", "0–255")

        st.markdown("**Flattened Vector Preview**")
        st.write(f"Label: {label}")
        flattened = result["flattened"]
        preview_text = "\n".join(
            f"Pixel_{i+1} = {int(flattened[i])}" for i in range(25)
        )
        st.code(preview_text, language="text")

st.divider()
st.caption(
    "Module 1 of the Handwritten Digit Recognition Platform — "
    "Data Collection Application"
)

import tempfile
import os
import io as _io
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import streamlit as st

st.set_page_config(page_title="Sagittal CT Viewer", layout="wide")
st.title("Sagittal CT Viewer")
st.write("Upload one or more `.nii.gz` / `.nii` files to visualize sagittal slices side by side.")

uploaded_files = st.file_uploader(
    "Upload CT / label volumes",
    type=None,
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Please upload at least one NIfTI file to begin.")
    st.stop()


@st.cache_data
def load_volume(file_bytes, suffix):
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        nii = nib.load(tmp_path)
        vol = nii.get_fdata()
        if vol.ndim == 4:
            vol = np.squeeze(vol, axis=0)
        return vol
    finally:
        os.unlink(tmp_path)


file_sig = tuple((uf.name, uf.size) for uf in uploaded_files)

if "file_sig" not in st.session_state or st.session_state.file_sig != file_sig:
    st.session_state.file_sig = file_sig
    st.session_state.volumes = {}
    for idx, uf in enumerate(uploaded_files):
        raw = uf.getvalue()
        suffix = ".nii.gz" if uf.name.endswith(".gz") else ".nii"
        vol = load_volume(raw, suffix)
        default_name = uf.name if len(uploaded_files) == 1 else f"{uf.name} (#{idx + 1})"
        st.session_state.volumes[f"{idx}_{uf.name}"] = (default_name, vol)
        st.success(f"Loaded `{default_name}` — shape {vol.shape}")

volumes = st.session_state.volumes

st.subheader("Plot titles")
if "custom_names" not in st.session_state:
    st.session_state.custom_names = {}

custom_names = {}
cols = st.columns(min(len(volumes), 3))
for i, key in enumerate(volumes):
    default_name, vol = volumes[key]
    if key not in st.session_state.custom_names:
        st.session_state.custom_names[key] = default_name
    with cols[i % len(cols)]:
        custom_names[key] = st.text_input(
            f"Title #{i + 1}",
            value=st.session_state.custom_names[key],
            key=f"name_{key}",
            on_change=lambda k=key: st.session_state.custom_names.update(
                {k: st.session_state[f"name_{k}"]}
            ),
        )

# Determine the number of sagittal slices (axis 0)
min_slices = min(vol.shape[0] for _, vol in volumes.values())

# Build a high-contrast colormap for label volumes
distinct_colors = [
    "#000000", "#e6194b", "#3cb44b", "#ffe119", "#0082c8",
    "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#d2f53c",
    "#fabebe", "#008080", "#e6beff", "#aa6e28", "#fffac8",
    "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000080",
    "#808080", "#ffffff", "#789ecf", "#bfefff", "#9a6324",
]
max_label = int(max(vol.max() for _, vol in volumes.values()))
while len(distinct_colors) <= max_label:
    distinct_colors.append("#%06x" % (hash(len(distinct_colors)) % 0xFFFFFF))
cmap = ListedColormap(distinct_colors[: max_label + 1])
cmap.set_bad(color="black")


def plot_labels(frame, ax, show_centroids=True, show_labels=True,
                centroid_size=8, text_size=5):
    if not show_centroids and not show_labels:
        return
    centroids = {}
    for label_val in np.unique(frame):
        if label_val == 0:
            continue
        coords = np.argwhere(frame == label_val)
        if len(coords) > 0:
            centroid_y, centroid_x = np.mean(coords, axis=0)
            centroids[label_val] = (centroid_x, centroid_y)
    for label_val, (x, y) in centroids.items():
        if show_centroids:
            ax.plot(x, y, "o", color="white", markersize=centroid_size,
                    markeredgecolor="black")
        if show_labels:
            ax.text(x + 5, y + 5, str(int(label_val)), color="white",
                    fontsize=text_size, weight="bold", ha="left", va="top")


col1, col2 = st.columns([1, 1])
with col1:
    show_centroids = st.checkbox("Show centroids", value=True)
with col2:
    show_labels = st.checkbox("Show labels", value=True)

col3, col4 = st.columns([1, 1])
with col3:
    centroid_size = st.slider("Centroid size", 1, 20, 8)
with col4:
    text_size = st.slider("Text size", 1, 20, 5)

slice_idx = st.slider("Sagittal slice", 0, min_slices - 1, min_slices // 2)

render = st.button("Render")

if render:
    n = len(volumes)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig_w = 5 * ncols
    fig_h = 6 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    axes_flat = axes.flatten()

    for i, (key, (display_name, vol)) in enumerate(volumes.items()):
        ax = axes_flat[i]
        sag = vol[slice_idx, :, :].T
        masked = np.ma.masked_where(sag == 0, sag)
        ax.imshow(masked, cmap=cmap, origin="lower", vmin=0, vmax=max_label)
        plot_labels(sag, ax, show_centroids=show_centroids, show_labels=show_labels,
                    centroid_size=centroid_size, text_size=text_size)
        ax.set_title(custom_names[key], fontsize=11)
        ax.axis("off")

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(f"Sagittal slice x={slice_idx}", fontsize=14)
    plt.tight_layout()

    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    display_width = min(ncols * 400, 1200)
    _, img_col, _ = st.columns([1, 3, 1])
    with img_col:
        st.image(buf, width=display_width)
else:
    _, info_col, _ = st.columns([1, 3, 1])
    with info_col:
        st.info("Click **Render** to generate the sagittal view.")

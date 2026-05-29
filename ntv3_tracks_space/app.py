import csv
import os
import tempfile
import time
import uuid
from pathlib import Path

import gradio as gr
import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from bigwig_export import _softmax_last, create_bigwig_zip
from ntv3_tracks_pipeline import (
    ASSEMBLY_TO_SPECIES,
    BED_ELEMENT_COLORS,
    SPECIES_WITH_COORDINATE_SUPPORT,
    load_ntv3_tracks_pipeline,
)

matplotlib.use("Agg")

# -----------------------------
# Env / auth
# -----------------------------
MODEL_ID = os.environ.get("MODEL_ID", "InstaDeepAI/NTv3_650M_post")
DEFAULT_SPECIES = os.environ.get("DEFAULT_SPECIES", "human")
HF_TOKEN = (
    os.environ.get("NTV3_HF_TOKEN")
    or os.environ.get("HF_TOKEN")
    or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
)
if HF_TOKEN is None:
    raise RuntimeError(
        "Missing Hugging Face token. Set NTV3_HF_TOKEN as a Space Secret."
    )

PLOT_TARGET_POINTS = int(os.environ.get("PLOT_TARGET_POINTS", "1500"))
SEARCH_MAX_RESULTS = int(os.environ.get("SEARCH_MAX_RESULTS", "20"))
MAX_SEQUENCE_SIZE = 1_048_576  # 1MB in bytes - maximum allowed sequence input size

# -----------------------------
# Load pipeline (reloadable)
# -----------------------------
pipe = None
current_model_id = MODEL_ID


def load_pipeline(model_id: str, species: str = DEFAULT_SPECIES):
    """Load or reload the pipeline with a new model."""
    global pipe, current_model_id
    pipe = load_ntv3_tracks_pipeline(
        model=model_id,
        token=HF_TOKEN,
        device="cpu",  # Prevents model.to(cuda) during import
        default_species=species,
        verbose=False,
    )
    current_model_id = model_id
    return pipe


# Load initial pipeline
load_pipeline(MODEL_ID, DEFAULT_SPECIES)


# -----------------------------
# Helpers
# -----------------------------

_t0 = None
_tlast = None


def tprint(msg: str):
    "Function to print timing information"
    global _t0, _tlast
    if _t0 is None:
        _t0 = _tlast = time.perf_counter()

    # CUDA ops are async → synchronize to get real timings
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    now = time.perf_counter()
    print(f"[timing] {msg}: {now - _tlast:.3f}s (total {now - _t0:.3f}s)")
    _tlast = now


# GPU decorator
try:
    import spaces
    gpu = spaces.GPU
except Exception:
    def gpu(*args, **kwargs):
        """GPU decorator placeholder when spaces module is not available."""
        def wrap(fn):
            return fn
        return wrap


def _global_stride(length: int, target: int) -> int:
    if target <= 0 or length <= target:
        return 1
    return int(np.ceil(length / target))


def _make_tracks_figure(
    x: np.ndarray, series: list[tuple[str, np.ndarray]], region: str = ""
):
    """Create an interactive plotly figure with multiple tracks."""
    if not series:
        raise gr.Error("Nothing to plot (no tracks/elements selected).")

    n = len(series)

    # Adjust vertical spacing based on number of tracks
    # More spacing when fewer tracks to prevent title overlap
    if n <= 2:
        vertical_spacing = 0.15  # More space for 1-2 tracks
    elif n <= 4:
        vertical_spacing = 0.08  # Moderate space for 3-4 tracks
    else:
        vertical_spacing = 0.04  # Tighter spacing for many tracks

    # Create subplots with shared x-axis
    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=vertical_spacing,
        subplot_titles=[title for title, _ in series],
    )

    # Define color schemes
    bigwig_color = "#4A90E2"  # Blue

    for i, (title, y) in enumerate(series, 1):
        # Determine color based on track type
        if title in BED_ELEMENT_COLORS:
            color = BED_ELEMENT_COLORS[title]
        else:
            color = bigwig_color

        # Convert color to rgba for fill
        rgba = mcolors.to_rgba(color)
        rgba_str = (
            f"rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, 0.3)"
        )

        # Add filled area (fill_between equivalent)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=title,
                line={"color": color, "width": 1.5},
                fill="tozeroy",
                fillcolor=rgba_str,
                hovertemplate=f"<b>{title}</b><br>"
                + "Position: %{x}<br>"
                + "Value: %{y:.4f}<extra></extra>",
                showlegend=False,
            ),
            row=i,
            col=1,
        )

    # Adjust height and margins based on number of tracks
    # More height per track when fewer tracks to accommodate titles
    if n <= 2:
        height_per_track = 200  # More height for 1-2 tracks
        top_margin = 60  # More top margin for titles
    elif n <= 4:
        height_per_track = 170  # Moderate height for 3-4 tracks
        top_margin = 50
    else:
        height_per_track = 150  # Standard height for many tracks
        top_margin = 40

    # Update layout for better appearance
    fig.update_layout(
        height=height_per_track * n,  # Adjust height based on number of tracks
        width=1200,
        margin={"l": 80, "r": 20, "t": top_margin, "b": 60},
        hovermode="x unified",  # Show all values at same x position
        template="plotly_white",
        modebar={
            "activecolor": "#7dd3fc",  # Blue color for active/hovered buttons
            "bgcolor": "rgba(255, 255, 255, 0.9)",
            "color": "#7dd3fc",  # Blue color for buttons
            "orientation": "v",
        },
    )

    # Update y-axes to remove ticks and improve appearance
    for i in range(1, n + 1):
        fig.update_yaxes(
            showticklabels=False,
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)",
            row=i,
            col=1,
        )

    # Update x-axis on the last subplot with region label
    xaxis_title = region if region else "Genomic position / index"
    fig.update_xaxes(
        title_text=xaxis_title,
        showgrid=True,
        gridcolor="rgba(0,0,0,0.1)",
        row=n,
        col=1,
    )

    return fig


# Cache track lists per species so search is instant after first load
_BIGWIG_CACHE: dict[str, list[str]] = {}

# Cache for track metadata (track_id -> display_name)
_TRACK_METADATA_CACHE: dict[str, str] = {}


def _load_track_metadata() -> dict[str, str]:
    """Load track metadata from CSV and create display name mapping."""
    if _TRACK_METADATA_CACHE:
        return _TRACK_METADATA_CACHE

    csv_path = Path(__file__).parent / "data" / "functional_tracks_metadata.csv"
    if not csv_path.exists():
        return {}

    metadata = {}
    try:
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                track_id = row["file_id"]
                tissue = row.get("tissue", "").strip()
                assay = row.get("assay", "").strip()
                experiment_target = row.get("experiment_target", "").strip()
                biosample_type = row.get("biosample_type", "").strip()
                strand = row.get("strand", "").strip()

                # Build display name from available fields
                parts = []
                # if biosample_type and biosample_type != "tissue":
                #     parts.append(biosample_type)
                if tissue:
                    parts.append(tissue)
                if assay:
                    # For RNA-seq, include strand information if available
                    if strand:
                        if strand == "plus":
                            strand = "+"
                        elif strand == "minus":
                            strand = "-"
                        parts.append(f"{assay} {strand}")
                    else:
                        parts.append(assay)
                if experiment_target and experiment_target not in ("none", "RNA-seq"):
                    parts.append(experiment_target)

                if parts:
                    display_name = " ".join(parts)
                else:
                    display_name = track_id  # Fallback to ID if no metadata

                metadata[track_id] = display_name
    except Exception as e:
        print(f"Warning: Could not load track metadata: {e}")
        return {}

    _TRACK_METADATA_CACHE.update(metadata)
    return metadata


def _get_track_display_name(track_id: str) -> str:
    """Get display name for a track ID, or return the ID if not found."""
    metadata = _load_track_metadata()
    return metadata.get(track_id, track_id)


def _format_track_for_display(track_id: str) -> str:
    """Format track ID for display: 'display_name (track_id)'."""
    display_name = _get_track_display_name(track_id)
    if display_name == track_id:
        return track_id  # No metadata available, just show ID
    return f"{display_name} ({track_id})"


def _extract_track_id(display_value: str) -> str:
    """Extract track ID from display format or return as-is."""
    if " (" in display_value and display_value.endswith(")"):
        # Extract track_id from format "display_name (track_id)"
        return display_value.rsplit(" (", 1)[1][:-1]
    return display_value  # No parentheses, assume it's already just the ID


def _get_bigwig_names(species: str) -> list[str]:
    if species not in _BIGWIG_CACHE:
        _BIGWIG_CACHE[species] = pipe.available_bigwig_track_names(species)
    return _BIGWIG_CACHE[species]


def _get_bed_element_names(species: str) -> list[str]:
    """Get BED element names available for a given species (filtered by training data)."""
    if pipe is None:
        return []
    try:
        return pipe.available_bed_element_names(species)
    except (ValueError, AttributeError):
        return []


def _format_bed_element_for_display(element_name: str) -> str:
    """Format BED element name for display: replace underscores with spaces and capitalize."""
    return element_name.replace("_", " ").title()


def _has_bigwigs(species: str) -> bool:
    """Check if a species has BigWig tracks available in the current model."""
    try:
        tracks = _get_bigwig_names(species)
        return len(tracks) > 0
    except (ValueError, AttributeError):
        # Species not in config or pipeline not loaded
        return False


def _get_species_with_bigwigs() -> set[str]:
    """Get set of species that have BigWig tracks available in the current model."""
    if pipe is None:
        return set()

    species_with_bigwigs = set()
    for species in ASSEMBLY_TO_SPECIES.values():
        if _has_bigwigs(species):
            species_with_bigwigs.add(species)
    return species_with_bigwigs


def _rank_search(query: str, names: list[str], limit: int) -> list[str]:
    """
    Return up to `limit` candidate track IDs matching `query` using a fast,
    low-overhead ranking suitable for very large `names` lists.

    Matching & ranking rules:
      1) Case-insensitive match.
      2) Items whose ID *starts with* the query are ranked first.
      3) Remaining items that merely *contain* the query are ranked after.
      4) Results preserve the original relative order within each group
         (stable w.r.t. the input `names` order).
      5) If `query` is empty/whitespace, returns an empty list to avoid
         flooding the UI with a huge default list.

    Notes:
      - `limit` only caps the number of returned results; it does not prevent
        short queries (e.g. "E") from producing many matches—if you want that,
        add a minimum query length check (e.g. `if len(q) < 2: return []`).
      - Time complexity is O(len(names)) per call.
    """
    q = (query or "").strip().lower()
    if not q:
        return []  # don’t spam a giant default list

    starts = []
    contains = []

    for n in names:
        nl = n.lower()
        if nl.startswith(q):
            starts.append(n)
        elif q in nl:
            contains.append(n)

    out = starts + contains
    return out[:limit]


def search_bigwigs(species: str, query: str, current_selected: list[str]):
    """Search BigWig tracks and return formatted display names."""
    # Handle None or empty query
    if query is None:
        query = ""
    query_stripped = query.strip()

    # If query is empty, return empty results immediately (don't show all tracks)
    if not query_stripped:
        displayed_selected = current_selected or []
        show_selected = bool(displayed_selected)
        return (
            gr.update(
                choices=[], value=[], interactive=True
            ),  # empty results, explicitly clear checked state
            gr.update(
                visible=show_selected,
                choices=displayed_selected,
                value=displayed_selected,
            ),  # show ALL selected tracks
        )

    names = _get_bigwig_names(species)
    # Search in both track IDs and display names
    metadata = _load_track_metadata()
    query_lower = query_stripped.lower()

    # Show selected tracks section if user is typing or has selections
    show_selected = bool(query_stripped) or bool(current_selected)

    # Show ALL selected tracks (not limited to 20)
    displayed_selected = current_selected or []

    # Extract track IDs from already selected tracks (to exclude them from results)
    selected_track_ids = set()
    if current_selected:
        selected_track_ids = {_extract_track_id(x) for x in current_selected}

    # Build list of (display_format, track_id) tuples for searching
    track_display_pairs = []
    for track_id in names:
        # Skip tracks that are already selected
        if track_id in selected_track_ids:
            continue
        display_name = metadata.get(track_id, track_id)
        display_format = _format_track_for_display(track_id)
        track_display_pairs.append((display_format, track_id, display_name))

    # Filter by query (search in display name, display format, and track_id)
    matching = []
    for display_format, track_id, display_name in track_display_pairs:
        if (
            query_lower in track_id.lower()
            or query_lower in display_name.lower()
            or query_lower in display_format.lower()
        ):
            matching.append(display_format)

    # Limit search results
    results = matching[:SEARCH_MAX_RESULTS]
    return (
        gr.update(
            choices=results, value=[], interactive=True
        ),  # results - limited to SEARCH_MAX_RESULTS, explicitly clear checked state
        gr.update(
            visible=show_selected, choices=displayed_selected, value=displayed_selected
        ),  # show ALL selected tracks
    )


def add_selected(current_selected: list[str], to_add: list[str]):
    """Add tracks to selected list, converting display format to track IDs if needed."""
    # Extract track IDs from current selection (in case they're in display format)
    cur_ids = [_extract_track_id(x) for x in (current_selected or [])]
    cur_display = [_format_track_for_display(tid) for tid in cur_ids]

    # Extract track IDs from items to add
    to_add_ids = [_extract_track_id(x) for x in (to_add or [])]

    # Add new track IDs
    for tid in to_add_ids:
        if tid not in cur_ids:
            cur_ids.append(tid)
            cur_display.append(_format_track_for_display(tid))

    # Show ALL selected tracks (no limit)
    return gr.update(choices=cur_display, value=cur_display)  # show all selected tracks


def remove_selected(current_selected: list[str], to_remove: list[str]):
    """Remove tracks from selected list."""
    cur = [x for x in (current_selected or []) if x not in set(to_remove or [])]
    # Show ALL remaining selected tracks (no limit)
    show_selected = bool(cur)
    return gr.update(choices=cur, value=cur, visible=show_selected)


def reset_on_species_change(species: str):
    """Reset search and selected tracks when species changes."""
    # Clear results + selected when species changes (avoids mismatched IDs)
    try:
        track_ids = _get_bigwig_names(species)  # warms cache if available
        # Format available tracks for display
        formatted_tracks = [_format_track_for_display(tid) for tid in track_ids]

        # Get default tracks for this species (filter to what's available)
        default_track_ids = [tid for tid in DEFAULT_BIGWIG_TRACKS if tid in track_ids]
        default_formatted = [
            _format_track_for_display(tid) for tid in default_track_ids
        ]

        # Show selected tracks section if there are default tracks
        show_selected = bool(default_formatted)

        return (
            gr.update(value=""),  # query textbox
            gr.update(choices=[], value=[]),  # results list
            gr.update(
                choices=formatted_tracks, value=default_formatted, visible=show_selected
            ),  # selected list with defaults
        )
    except (ValueError, AttributeError):
        # Species doesn't have bigwigs, that's okay
        return (
            gr.update(value=""),  # query textbox
            gr.update(choices=[], value=[]),  # results list
            gr.update(choices=[], value=[], visible=False),  # selected list (hidden)
        )


# -----------------------------
# Predict
# -----------------------------
@gpu
def predict(
    seq: str,
    species: str,
    chrom: str,
    start: int,
    end: int,
    input_type: str,
    bigwig_selected: list[str],
    bed_elements: list[str],
):
    """Run prediction and return figure with tracks."""
    tprint("start")

    # Debug: verify species is being passed
    if not species:
        raise gr.Error("Species parameter is missing. Please select a species.")

    # Extract track IDs from display format if needed
    bigwig_selected = [_extract_track_id(tid) for tid in bigwig_selected]

    # Determine if using coordinates based on input_type radio button
    use_coords = input_type == "Use genomic coordinates"

    if use_coords:
        # Check if this species supports coordinate-based fetching
        if species not in SPECIES_WITH_COORDINATE_SUPPORT:
            supported = ", ".join(sorted(SPECIES_WITH_COORDINATE_SUPPORT))
            raise gr.Error(
                f"Species '{species}' does not support coordinate-based sequence "
                f"fetching. Please provide a DNA sequence directly or use one of "
                f"the supported species: {supported}"
            )
        if not chrom:
            raise gr.Error("chrom is required when use_coords=True")
        if start is None or end is None or int(end) <= int(start):
            raise gr.Error("start/end must be set and end > start when use_coords=True")
        
        # Check sequence size before fetching from API: max 1MB
        # Each base pair is typically 1 byte, so check region length
        region_length = int(end) - int(start)
        if region_length > MAX_SEQUENCE_SIZE:
            raise gr.Error(
                f"Requested genomic region is too large ({region_length:,} base pairs). "
                f"Maximum allowed size is {MAX_SEQUENCE_SIZE:,} base pairs (1MB). "
                f"Please select a smaller region."
            )
        
        inputs = {
            "chrom": chrom,
            "start": int(start),
            "end": int(end),
            "species": species,
        }
    else:
        if not seq or not seq.strip():
            raise gr.Error("seq is required when use_coords=False")
        seq_stripped = seq.strip()
        # Check sequence size: max 1MB
        # Each character is typically 1 byte, so check length
        if len(seq_stripped) > MAX_SEQUENCE_SIZE:
            raise gr.Error(
                f"Sequence input is too large ({len(seq_stripped):,} characters). "
                f"Maximum allowed size is {MAX_SEQUENCE_SIZE:,} characters (1MB). "
                f"Please use a shorter sequence or use genomic coordinates instead."
            )
        inputs = {"seq": seq_stripped, "species": species}

    # Verify species is in inputs before calling pipeline
    if "species" not in inputs:
        input_keys = list(inputs.keys())
        raise gr.Error(
            f"Internal error: species not found in inputs dict. "
            f"Inputs: {input_keys}"
        )

    tprint("inputs prepared")

    # move to GPU only once the ZeroGPU context is active
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # check where the model currently lives
    current = next(pipe.model.parameters()).device.type  # "cpu" or "cuda"
    # only move if needed
    if current != device:
        pipe.model.to(device)
        tprint(f"model moved to {device}")

    pipe.model.eval()
    print(f"Running on {next(pipe.model.parameters()).device}")
    tprint("model ready to run inference")

    # run inference
    out = pipe(inputs)

    tprint("inference completed")

    # optional: move back to CPU so you don’t rely on any persistent CUDA context
    # if device == "cuda":
    #     pipe.model.to("cpu")

    bw_names = out.bigwig_track_names or []
    bw = out.bigwig_tracks_logits
    bed_names = out.bed_element_names or []
    bed_logits = out.bed_tracks_logits

    # Check if we have any tracks/elements to plot
    has_bigwigs = bw is not None and len(bw_names) > 0
    has_bed = bed_logits is not None and len(bed_names) > 0

    if not has_bigwigs and not has_bed:
        raise gr.Error(
            "No BigWig tracks or BED elements available for this species "
            "in the current model."
        )

    if not has_bigwigs and bigwig_selected:
        raise gr.Error(
            "No BigWig tracks available for this species, but BigWig tracks "
            "were selected. Please deselect BigWig tracks or choose a "
            "different species."
        )

    # Defaults if user picked none
    if has_bigwigs and not bigwig_selected:
        # Filter to only include tracks that are available for this species/assembly
        bigwig_selected = [tid for tid in DEFAULT_BIGWIG_TRACKS if tid in bw_names]

    if (not bed_elements) and bed_names:
        default_bed_elements = ["protein_coding_gene", "exon", "intron"]
        # Filter to only include elements that are available
        bed_elements = [elem for elem in default_bed_elements if elem in bed_names]

    # Validate (important for API usage)
    if has_bigwigs and bigwig_selected:
        missing_tracks = [t for t in bigwig_selected if t not in bw_names]
        if missing_tracks:
            raise gr.Error(f"Unknown BigWig track id(s): {missing_tracks}")

    if bed_elements:
        missing_elems = [e for e in bed_elements if e not in bed_names]
        if missing_elems:
            raise gr.Error(f"Unknown BED element(s): {missing_elems}")

    # Determine sequence length from available data
    if has_bigwigs:
        seq_length = bw.shape[0]
    elif has_bed:
        seq_length = bed_logits.shape[0]
    else:
        raise gr.Error("No data available for plotting.")

    stride = _global_stride(seq_length, PLOT_TARGET_POINTS)

    x0 = int(out.pred_start or 0)
    x1 = int(out.pred_end or (x0 + seq_length))
    x = np.linspace(x0, x1, num=seq_length, endpoint=False)[::stride]

    series: list[tuple[str, np.ndarray]] = []

    # Add BigWig tracks if available and selected
    if has_bigwigs and bigwig_selected:
        for tid in bigwig_selected:
            idx = bw_names.index(tid)
            # Use clean display name instead of track ID
            display_name = _get_track_display_name(tid)
            series.append((display_name, bw[:, idx][::stride].astype(float)))

    # Add BED elements if available and selected
    if bed_logits is not None and bed_elements:
        probs = _softmax_last(bed_logits)
        for ename in bed_elements:
            display_name = ename.replace("_", " ").lower()
            eidx = bed_names.index(ename)
            series.append((display_name, probs[:, eidx, 1][::stride].astype(float)))

    tprint("figure data processed created")

    # Build region string for x-axis label
    region = (
        f"{out.chrom}:{out.pred_start}-{out.pred_end}" if out.chrom else f"{x0}-{x1}"
    )
    if out.assembly:
        region += f" ({out.assembly})"

    fig = _make_tracks_figure(x, series, region=region)
    tprint("figure created")

    meta = {
        "model_id": current_model_id,
        "species": out.species,
        "assembly": out.assembly,
        "chrom": out.chrom,
        "pred_start": out.pred_start,
        "pred_end": out.pred_end,
        "bigwig_selected": bigwig_selected,
        "bed_selected": bed_elements,
        "plot_stride": stride,
        "plot_target_points": PLOT_TARGET_POINTS,
    }

    return (
        gr.update(visible=True),  # predictions_heading
        gr.update(visible=True),  # predictions_note
        gr.update(value=fig, visible=True),  # plot
        gr.update(visible=True),  # download_bigwig_btn
        # meta,
        out,
        bigwig_selected,
        bed_elements,
    )


# -----------------------------
# UI (keep your download icon setup)
# -----------------------------
# Load CSS from external file
CSS_PATH = Path(__file__).parent / "style.css"
CSS = CSS_PATH.read_text() if CSS_PATH.exists() else ""

JS = """
// Remove blue backgrounds from footer buttons after page loads
function removeFooterButtonBackgrounds() {
    // Target all buttons and links in footer
    const footer = document.querySelector('footer');
    if (footer) {
        const buttons = footer.querySelectorAll('button, a[role="button"], a[class*="button"]');
        buttons.forEach(btn => {
            btn.style.setProperty('background', 'transparent', 'important');
            btn.style.setProperty('background-color', 'transparent', 'important');
            btn.style.setProperty('border', '1px solid rgba(125, 211, 252, 0.2)', 'important');
        });
    }
}

// Run on page load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', removeFooterButtonBackgrounds);
} else {
    removeFooterButtonBackgrounds();
}

// Also run after a delay to catch dynamically loaded content
setTimeout(removeFooterButtonBackgrounds, 100);
setTimeout(removeFooterButtonBackgrounds, 500);
setTimeout(removeFooterButtonBackgrounds, 1000);

// Use MutationObserver to watch for dynamically added footer buttons
const observer = new MutationObserver(() => {
    removeFooterButtonBackgrounds();
});

// Observe changes to the footer
const footer = document.querySelector('footer');
if (footer) {
    observer.observe(footer, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ['style', 'class']
    });
}

// Also observe the entire document for footer additions
observer.observe(document.body, {
    childList: true,
    subtree: true
});
"""

# BED list is small enough to keep as dropdown
# Filter by default species to show only elements available for training
_init_bed = pipe.available_bed_element_names(DEFAULT_SPECIES)

# Default BigWig tracks
DEFAULT_BIGWIG_TRACKS = [
    "ENCSR056HPM",  # K562 RNA-seq
    "ENCSR921NMD",  # K562 DNAse
    "ENCSR000DWD",  # K562 H3k4me3
    "ENCSR000AKO",  # K562 CTCF
    "ENCSR561FEE_P",  # HepG2 RNA-seq
    "ENCSR000EJV",  # HepG2 DNAse
    "ENCSR000AMP",  # HepG2 H3k4me3
    "ENCSR000BIE",  # HepG2 CTCF
]

# Default BED elements
DEFAULT_BED_ELEMENTS = ["protein_coding_gene", "exon", "intron"]

# Get available BigWig tracks for default species and filter defaults
_init_bigwig = _get_bigwig_names(DEFAULT_SPECIES)
_init_bigwig_selected_ids = [
    tid for tid in DEFAULT_BIGWIG_TRACKS if tid in _init_bigwig
]
# Format for display
_init_bigwig_selected = [
    _format_track_for_display(tid) for tid in _init_bigwig_selected_ids
]

# Filter default BED elements to only those available
_init_bed_selected = [elem for elem in DEFAULT_BED_ELEMENTS if elem in _init_bed]

# Format BED elements for display: use tuples (display_name, value) for dropdown
_init_bed_choices = [
    (_format_bed_element_for_display(elem), elem) for elem in _init_bed
]
_init_bed_selected_values = _init_bed_selected  # Keep original values for selection

# Default coordinates per species
DEFAULT_COORDS = {
    "human": {"chrom": "chr19", "start": 6_700_000, "end": 6_831_072},
    "mouse": {"chrom": "chr1", "start": 9_880_168, "end": 10_142_312},
    "drosophila_melanogaster": {"chrom": "chr2L", "start": 6_700_000, "end": 6_831_072},
    "arabidopsis_thaliana": {"chrom": "chr1", "start": 13_135_095, "end": 13_397_239},
}

# Get default coordinates for default species
_default_coords = DEFAULT_COORDS.get(DEFAULT_SPECIES, DEFAULT_COORDS["human"])


# Format species names for display (replace underscores with spaces, capitalize first letter)
def _format_species_name(species: str) -> str:
    """Format species name for display."""
    return species.replace("_", " ").capitalize()


# Get all available species and format them
_all_species = sorted(ASSEMBLY_TO_SPECIES.values())
_all_species_formatted = [_format_species_name(s) for s in _all_species]
_all_species_list = ", ".join(_all_species_formatted)

# Get species with BigWig tracks
_species_with_bigwigs = _get_species_with_bigwigs()
_bigwig_species_formatted = sorted(
    [_format_species_name(s) for s in _species_with_bigwigs]
)
_bigwig_species_list = (
    ", ".join(_bigwig_species_formatted)
    if _bigwig_species_formatted
    else "None (BED elements only)"
)

with gr.Blocks(title="NTv3 Tracks Demo") as demo:
    gr.Markdown(
        f"""
<div class="intro-hero">

<div class="intro-title">
  <h1>🧬 NTv3 Tracks Demo</h1>
  <p class="intro-subtitle">
    Predict and visualize functional genomics signals directly from DNA using
    <strong>Nucleotide Transformer v3</strong>.
  </p>
  
  <div class="intro-pillrow">
    <span class="intro-pill">🧬 Functional genomics</span>
    <span class="intro-pill">📊 BigWig tracks</span>
    <span class="intro-pill">🎯 Genome annotation</span>
    <span class="intro-pill">🌍 Multi-species</span>
    <span class="intro-pill">⚡ Interactive visualization</span>
  </div>
</div>

<div class="intro-grid">
  <div class="intro-card">
    <h3>1) Provide input</h3>
    <ul>
      <li>Select a <strong>model</strong> and <strong>species</strong></li>
      <li>Use genomic coordinates (chrom, start, end), <em>or</em></li>
      <li>Paste a DNA sequence</li>
    </ul>
  </div>

  <div class="intro-card">
    <h3>2) Choose signals</h3>
    <ul>
      <li>Search & select <strong>BigWig functional tracks</strong>
        (RNA-seq, ChIP-seq, DNase…)</li>
      <li>Select <strong>BED genome annotation elements</strong>
        (exons, introns, promoters…)</li>
    </ul>
  </div>

  <div class="intro-card">
    <h3>3) Explore</h3>
    <ul>
      <li>View stacked tracks across the region</li>
      <li>Compare multiple tracks side-by-side</li>
      <li>Download plot and BigWig files</li>
    </ul>
  </div>
</div>

<div class="intro-tip">
  <span class="intro-tip-icon">💡</span>
  <span><strong>Tip:</strong> The demo includes default settings that you can use
    to get started, taking ~ 15 seconds to run for the example on human.</span>
</div>

<div class="intro-species-info">
  <strong>Available species:</strong> {_all_species_list}<br>
  <br>
  <strong>Species with functional tracks:</strong> {_bigwig_species_list}
</div>

</div>
""",
        elem_id="intro_markdown",
    )

    gr.Markdown("# Select NTv3 post-trained model")

    # Model display names (without InstaDeepAI/ prefix) and their full IDs
    MODEL_OPTIONS = {
        "NTv3 650M (post)": "InstaDeepAI/NTv3_650M_post",
        "NTv3 100M (post)": "InstaDeepAI/NTv3_100M_post",
    }

    # Reverse mapping: full ID -> display name
    MODEL_ID_TO_DISPLAY = {v: k for k, v in MODEL_OPTIONS.items()}

    # Get display name for current model
    current_display_name = MODEL_ID_TO_DISPLAY.get(current_model_id, "NTv3 100M (pos)")

    model_selector = gr.Dropdown(
        choices=list(MODEL_OPTIONS.keys()),
        value=current_display_name,
        label="Model",
    )

    model_status = gr.Markdown("", visible=False)

    gr.Markdown("# Input DNA sequence")

    # Get all available species from the pipeline and format for display
    all_species = sorted(ASSEMBLY_TO_SPECIES.values())
    # Format choices as (display_name, value) tuples so dropdown shows formatted names
    # but returns actual species values
    species_choices = [(_format_species_name(s), s) for s in all_species]

    species = gr.Dropdown(
        choices=species_choices,
        value=DEFAULT_SPECIES,
        label="Species",
    )

    # Radio buttons for input type selection
    is_supported_default = DEFAULT_SPECIES in SPECIES_WITH_COORDINATE_SUPPORT
    initial_input_type = (
        "Use genomic coordinates" if is_supported_default else "Enter DNA sequence"
    )
    input_type = gr.Radio(
        choices=["Use genomic coordinates", "Enter DNA sequence"],
        value=initial_input_type,
        label="Input method",
        visible=is_supported_default,  # Only show if species supports coordinates
    )

    # Coordinates section - visible only when "Use genomic coordinates" is selected
    with gr.Group(
        visible=is_supported_default
        and initial_input_type == "Use genomic coordinates",
        elem_id="coords_group",
    ) as coords_group:
        gr.Markdown(
            "**Genomic coordinates** (supported species: "
            + ", ".join(sorted(SPECIES_WITH_COORDINATE_SUPPORT))
            + ")"
        )
        with gr.Row():
                # chrom = gr.Textbox(
                #     label="Chromosome", value=_default_coords["chrom"], elem_id="chromosome_input"
                # )
                chrom = gr.Dropdown(
                    label="Chromosome",
                    choices=[],                 # no predefined list
                    value=_default_coords["chrom"],
                    allow_custom_value=True,    # user can type anything (e.g. chr19, scaffold_123)
                    filterable=True,            # enables typing/search UI
                    elem_id="chromosome_input",
                )
                start = gr.Number(
                    label="Start", value=_default_coords["start"], precision=0, elem_id="start_input"
                )
                end = gr.Number(
                    label="End", value=_default_coords["end"], precision=0, elem_id="end_input"
                )

    # DNA sequence section - visible only when "Enter DNA sequence" is selected
    # Using Textbox directly (not wrapped in Group) to avoid visual border/line
    seq = gr.Textbox(
        lines=4,
        label="Input DNA sequence",
        placeholder="ACGT...",
        visible=initial_input_type == "Enter DNA sequence",
        elem_id="dna_sequence_input",
    )

    def change_model(display_name: str, species: str):
        """Reload pipeline with new model."""
        try:
            # Convert display name to full model ID
            if display_name in MODEL_OPTIONS:
                model_id = MODEL_OPTIONS[display_name]
            else:
                # Fallback: assume it's already a model ID or custom value
                model_id = display_name

            load_pipeline(model_id, species)
            # Update available tracks/elements
            _get_bigwig_names(species)  # warm cache
            return gr.update(value="✅ Model loaded successfully"), gr.update(
                visible=True
            )
        except Exception as e:
            return gr.update(value=f"❌ Error loading model: {str(e)}"), gr.update(
                visible=True
            )

    model_selector.change(
        fn=change_model,
        inputs=[model_selector, species],
        outputs=[model_status, model_status],
    )

    gr.Markdown("# Select functional tracks")

    # Button to download tracks metadata
    def get_metadata_file_path():
        """Return path to metadata CSV file for download."""
        csv_path = Path(__file__).parent / "data" / "functional_tracks_metadata.csv"
        if csv_path.exists():
            return str(csv_path)
        return None

    metadata_file_path = get_metadata_file_path()
    download_metadata_btn = gr.Button(
        "📋 Download metadata for all functional tracks",
        variant="secondary",
        visible=metadata_file_path is not None,
    )
    metadata_download_file = gr.File(
        label="Tracks metadata",
        visible=False,
    )

    def download_metadata():
        """Return metadata file for download."""
        if metadata_file_path and Path(metadata_file_path).exists():
            return gr.update(value=metadata_file_path, visible=True)
        return gr.update(visible=False)

    download_metadata_btn.click(
        fn=download_metadata,
        inputs=[],
        outputs=[metadata_download_file],
    )

    bigwig_no_tracks_msg = gr.Markdown(
        "⚠️ No functional genomic tracks available for this species "
        "in the current model.",
        visible=False,
    )

    bigwig_query = gr.Textbox(
        label="Search functional tracks (auto-search while typing)",
        placeholder="Type to search… (e.g. heart DNAse-seq)",
    )

    bigwig_results = gr.CheckboxGroup(
        choices=[],
        label="Results (click to add to Selected)",
    )

    bigwig_selected = gr.CheckboxGroup(
        choices=_init_bigwig_selected,
        value=_init_bigwig_selected,
        label="Selected functional tracks (used for prediction)",
        visible=bool(
            _init_bigwig_selected
        ),  # Show if there are default tracks, otherwise hidden
    )

    with gr.Row(visible=True) as bigwig_buttons_row:
        bigwig_clear_btn = gr.Button("Clear search results")
        bigwig_remove_btn = gr.Button("Remove all selected")

    gr.Markdown("# Select genome annotation elements")

    bed_elements = gr.Dropdown(
        choices=_init_bed_choices,
        value=_init_bed_selected_values if _init_bed_selected_values else [],
        multiselect=True,
        label="Genome annotation elements (search + select)",
        elem_id="bed_elements_dropdown",
    )

    btn = gr.Button("Predict", elem_id="predict_btn")

    predictions_heading = gr.Markdown(
        "# NTv3 predictions for selected tracks and elements\n\n", visible=False
    )
    predictions_note = gr.Markdown(
        "Note: NTv3 predictions are for the 37.5% center of the input sequence.",
        visible=False,
    )

    plot = gr.Plot(label="", elem_id="tracks_plot", visible=False)

    # State to store prediction output and selections for BigWig export
    prediction_state = gr.State(value=None)
    bigwig_selected_state = gr.State(value=[])
    bed_elements_state = gr.State(value=[])

    download_bigwig_btn = gr.Button(
        "📥 Download tracks as BigWig files (ZIP)",
        variant="secondary",
        visible=False,
    )
    export_bigwig = gr.File(label="Download BigWig files", visible=False)

    # --- wiring (live search + auto-add) ---

    # Live search on every keystroke and when text changes (including deletion)
    bigwig_query.input(
        fn=search_bigwigs,
        inputs=[species, bigwig_query, bigwig_selected],
        outputs=[bigwig_results, bigwig_selected],
    )
    # Also trigger on change to catch deletions
    bigwig_query.change(
        fn=search_bigwigs,
        inputs=[species, bigwig_query, bigwig_selected],
        outputs=[bigwig_results, bigwig_selected],
    )

    # Helper function to get search results choices directly (without gr.update wrapper)
    def _get_search_results_choices(
        species: str, query: str, current_selected: list[str]
    ) -> list[str]:
        """Get search results choices as a list, excluding selected tracks."""
        if query is None:
            query = ""
        query_stripped = query.strip()

        if not query_stripped:
            return []

        names = _get_bigwig_names(species)
        metadata = _load_track_metadata()
        query_lower = query_stripped.lower()

        # Extract track IDs from already selected tracks
        selected_track_ids = set()
        if current_selected:
            selected_track_ids = {_extract_track_id(x) for x in current_selected}

        # Build and filter results
        matching = []
        for track_id in names:
            if track_id in selected_track_ids:
                continue
            display_name = metadata.get(track_id, track_id)
            display_format = _format_track_for_display(track_id)
            if (
                query_lower in track_id.lower()
                or query_lower in display_name.lower()
                or query_lower in display_format.lower()
            ):
                matching.append(display_format)

        return matching[:SEARCH_MAX_RESULTS]

    # Auto-add: whenever user checks items in results, add them to Selected,
    # then clear results selection (so it feels like "click to add")
    def _auto_add(
        selected_now: list[str],
        results_checked: list[str],
        current_query: str,
        current_results: list[str],
        current_species: str,
    ):
        """Add selected tracks and refresh search results."""
        upd = add_selected(selected_now, results_checked)
        show_selected = bool(upd["value"])

        # Get updated search results (excluding newly selected tracks)
        new_choices = _get_search_results_choices(
            current_species, current_query, upd["value"]
        )

        # Clear checked state by setting empty value
        fresh_update = gr.update(choices=new_choices, value=[])

        return gr.update(**upd, visible=show_selected), fresh_update

    # Use a wrapper that ensures results are cleared before updating
    def _auto_add_wrapper(
        selected_now: list[str],
        results_checked: list[str],
        current_query: str,
        current_results: list[str],
        current_species: str,
    ):
        """Wrapper to ensure results are cleared after adding tracks."""
        return _auto_add(
            selected_now,
            results_checked,
            current_query,
            current_results,
            current_species,
        )

    bigwig_results.change(
        fn=_auto_add_wrapper,
        inputs=[bigwig_selected, bigwig_results, bigwig_query, bigwig_results, species],
        outputs=[bigwig_selected, bigwig_results],
    )

    # Update selected tracks immediately when user unchecks items
    def _update_selected_tracks(
        selected_value: list[str], current_query: str, current_species: str
    ):
        """Update selected tracks when user checks/unchecks items directly."""
        # selected_value contains only the currently checked items
        # Update choices to match current selections
        # (unchecked items are removed)
        show_selected = bool(selected_value)

        # Also update search results to reflect new selection
        # (unchecked tracks can now appear in results)
        search_updates = search_bigwigs(current_species, current_query, selected_value)

        return (
            gr.update(
                choices=selected_value, value=selected_value, visible=show_selected
            ),  # Update selected tracks
            search_updates[0],  # Update search results
        )

    bigwig_selected.change(
        fn=_update_selected_tracks,
        inputs=[bigwig_selected, bigwig_query, species],
        outputs=[bigwig_selected, bigwig_results],
    )

    # Clear results list (handy when query is short)
    def _clear_results():
        return gr.update(choices=[], value=[]), gr.update(value="")

    bigwig_clear_btn.click(
        fn=_clear_results,
        inputs=[],
        outputs=[bigwig_results, bigwig_query],
    )

    # Remove: check items in Selected, then click Remove
    bigwig_remove_btn.click(
        fn=remove_selected,
        inputs=[bigwig_selected, bigwig_selected],
        outputs=[bigwig_selected],
    )

    species.change(
        fn=reset_on_species_change,
        inputs=[species],
        outputs=[bigwig_query, bigwig_results, bigwig_selected],
    )

    # Update coordinates visibility and values when species changes
    def update_on_species_change(species: str, input_type_val: str):
        """Update coordinates visibility and values when species changes."""
        is_supported = species in SPECIES_WITH_COORDINATE_SUPPORT
        has_bigwigs = _has_bigwigs(species)
        coords = DEFAULT_COORDS.get(species, DEFAULT_COORDS["human"])
        # Show coordinates only if species is supported AND input type is coordinates
        use_coords = input_type_val == "Use genomic coordinates"
        show_coords = is_supported and use_coords
        show_seq = not show_coords

        # Format available tracks for display if species has bigwigs
        formatted_tracks = []
        default_formatted = []
        show_selected_tracks = False
        
        if has_bigwigs:
            try:
                track_ids = _get_bigwig_names(species)
                formatted_tracks = [_format_track_for_display(tid) for tid in track_ids]
                default_track_ids = [tid for tid in DEFAULT_BIGWIG_TRACKS if tid in track_ids]
                default_formatted = [_format_track_for_display(tid) for tid in default_track_ids]
                show_selected_tracks = bool(default_formatted)
            except Exception:
                pass

        # Get BED elements available for this species
        bed_element_names = _get_bed_element_names(species)
        # Filter default BED elements to only those available for this species
        default_bed_selected = [
            elem for elem in DEFAULT_BED_ELEMENTS if elem in bed_element_names
        ]
        # Format BED elements for display: use tuples (display_name, value)
        bed_element_choices = [
            (_format_bed_element_for_display(elem), elem) for elem in bed_element_names
        ]

        return (
            gr.update(visible=show_coords, value=coords["chrom"]),
            gr.update(visible=show_coords, value=coords["start"]),
            gr.update(visible=show_coords, value=coords["end"]),
            gr.update(
                visible=is_supported,
                value="Use genomic coordinates"
                if is_supported
                else "Enter DNA sequence",
            ),  # Update input_type radio
            gr.update(visible=show_coords),  # Show/hide coords_group
            gr.update(visible=show_seq),  # Show/hide seq
            gr.update(
                visible=not has_bigwigs
            ),  # Show "no tracks" message if no bigwigs
            gr.update(
                visible=show_selected_tracks,
                choices=formatted_tracks,
                value=default_formatted,
            ),  # Show bigwig selection with defaults if available
            gr.update(visible=has_bigwigs),  # Show bigwig query if available
            gr.update(visible=has_bigwigs),  # Show bigwig results if available
            gr.update(visible=has_bigwigs),  # Show bigwig buttons if available
            gr.update(
                choices=bed_element_choices,
                value=default_bed_selected,
            ),  # Update BED elements dropdown with species-specific elements
        )

    # Update input type radio visibility and value when species changes
    def update_input_type_on_species_change(species: str):
        """Update input type radio when species changes."""
        is_supported = species in SPECIES_WITH_COORDINATE_SUPPORT
        # If species doesn't support coordinates, default to sequence input
        default_value = (
            "Use genomic coordinates" if is_supported else "Enter DNA sequence"
        )
        return gr.update(visible=is_supported, value=default_value)

    # Update input visibility when radio button changes
    def update_input_visibility(input_type_val: str, species: str):
        """Update input visibility when radio button changes."""
        is_supported = species in SPECIES_WITH_COORDINATE_SUPPORT
        use_coords = input_type_val == "Use genomic coordinates"
        show_coords = is_supported and use_coords
        
        return (
            gr.update(visible=show_coords),  # coords_group
            gr.update(visible=not show_coords),  # seq
            gr.update(visible=show_coords),  # chrom
            gr.update(visible=show_coords),  # start
            gr.update(visible=show_coords),  # end
        )

    species.change(
        fn=update_input_type_on_species_change,
        inputs=[species],
        outputs=[input_type],
    )

    species.change(
        fn=update_on_species_change,
        inputs=[species, input_type],
        outputs=[
            chrom,
            start,
            end,
            input_type,
            coords_group,
            seq,
            bigwig_no_tracks_msg,
            bigwig_selected,
            bigwig_query,
            bigwig_results,
            bigwig_buttons_row,
            bed_elements,
        ],
    )

    input_type.change(
        fn=update_input_visibility,
        inputs=[input_type, species],
        outputs=[coords_group, seq, chrom, start, end],
    )

    def show_prediction_ui():
        """Show prediction UI elements immediately when Predict is clicked."""
        return (
            gr.update(visible=True),  # predictions_heading
            gr.update(visible=True),  # predictions_note
            gr.update(visible=True),  # plot (shows progress bar)
            gr.update(visible=False),  # download_bigwig_btn (will be shown after prediction)
        )

    # Show UI elements immediately when button is clicked
    btn.click(
        fn=show_prediction_ui,
        inputs=[],
        outputs=[
            predictions_heading,
            predictions_note,
            plot,
            download_bigwig_btn,
        ],
    ).then(
        fn=predict,
        inputs=[
            seq,
            species,
            chrom,
            start,
            end,
            input_type,
            bigwig_selected,
            bed_elements,
        ],
        outputs=[
            predictions_heading,
            predictions_note,
            plot,
            download_bigwig_btn,
            # meta,
            prediction_state,
            bigwig_selected_state,
            bed_elements_state,
        ],
        api_name="predict",
    )

    def download_bigwig_zip(out, bw_selected, bed_selected):
        """Create and return BigWig zip file."""
        try:
            zip_path = create_bigwig_zip(out, bw_selected, bed_selected)
            return gr.update(value=zip_path, visible=True)
        except ImportError:
            raise gr.Error(
                "pyBigWig is required for BigWig export. "
                "Install with: pip install pyBigWig"
            )
        except Exception as exc:
            raise gr.Error(f"Error creating BigWig files: {str(exc)}")

    download_bigwig_btn.click(
        fn=download_bigwig_zip,
        inputs=[prediction_state, bigwig_selected_state, bed_elements_state],
        outputs=[export_bigwig],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        ssr_mode=False,
        show_error=True,
        allowed_paths=[tempfile.gettempdir()],
        css=CSS,
        js=JS,
    )

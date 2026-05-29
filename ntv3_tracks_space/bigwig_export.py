"""
BigWig export functionality for NTv3 tracks.
"""

import os
import tempfile
import uuid
import zipfile
from typing import TYPE_CHECKING

import numpy as np

try:
    import pyBigWig  # noqa: N816
except ImportError:
    pyBigWig = None  # noqa: N816

if TYPE_CHECKING:
    from ntv3_tracks_pipeline import NTv3TracksOutput


def _softmax_last(x: np.ndarray) -> np.ndarray:
    """Compute softmax over the last dimension."""
    x = x - x.max(axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=-1, keepdims=True)


def create_bigwig_zip(
    out: "NTv3TracksOutput",
    bigwig_selected: list[str],
    bed_elements: list[str],
) -> str:
    """
    Create BigWig files for selected tracks and save them in a zip file.

    Parameters
    ----------
    out : NTv3TracksOutput
        The prediction output from the pipeline.
    bigwig_selected : list[str]
        List of BigWig track IDs to export.
    bed_elements : list[str]
        List of BED element names to export.

    Returns
    -------
    str
        Path to the created zip file containing BigWig files.

    Raises
    ------
    ImportError
        If pyBigWig is not installed.
    ValueError
        If no predictions are available or no tracks are selected.
    """
    if pyBigWig is None:
        raise ImportError(
            "pyBigWig is required for BigWig export. Install with: pip install pyBigWig"
        )

    if out is None:
        raise ValueError("No predictions available. Please run a prediction first.")

    bw_names = out.bigwig_track_names or []
    bw_logits = out.bigwig_tracks_logits
    bed_names = out.bed_element_names or []
    bed_logits = out.bed_tracks_logits

    if bw_logits is None or not bw_names:
        raise ValueError("No BigWig tracks available in model output.")

    # Get genomic coordinates
    chrom = out.chrom
    if chrom is None:
        raise ValueError(
            "Chromosome information not available. Use genomic coordinates."
        )

    start = out.start
    end = out.end
    if start is None or end is None:
        raise ValueError("Start and end coordinates are required for BigWig export.")
    window_len = out.window_len or (end - start)

    # Calculate prediction region (center 37.5%)
    if out.pred_start is not None:
        pred_start = out.pred_start
    else:
        pred_start = start + int(window_len * 0.3125)

    # Create temporary directory for BigWig files
    tmpdir = tempfile.gettempdir()
    output_dir = os.path.join(tmpdir, f"bigwig_outputs_{uuid.uuid4().hex}")
    os.makedirs(output_dir, exist_ok=True)

    # Prepare track data list
    track_data_list = []

    # Add BigWig tracks
    for track_id in bigwig_selected:
        if track_id in bw_names:
            idx = bw_names.index(track_id)
            track_data_list.append(("bigwig", track_id, idx, None))

    # Add BED elements (as probabilities)
    if bed_logits is not None and bed_elements:
        probs = _softmax_last(bed_logits)
        for elem_name in bed_elements:
            if elem_name in bed_names:
                eidx = bed_names.index(elem_name)
                # Store as bed element with probability data
                track_data_list.append(("bed", elem_name, eidx, probs[:, eidx, 1]))

    if not track_data_list:
        raise ValueError("No tracks selected for export.")

    # Create BigWig files
    created_files = []
    for track_type, track_id, track_idx, bed_probs in track_data_list:
        if track_type == "bigwig":
            track_data = bw_logits[:, track_idx].astype(np.float32)
            display_name = track_id
        else:  # bed
            if bed_probs is None:
                continue
            track_data = bed_probs.astype(np.float32)
            display_name = track_id

        # Clean filename
        clean_name = display_name.replace(" ", "_").replace("/", "_").replace("-", "_")
        bw_filename = os.path.join(output_dir, f"{clean_name}.bw")

        # Create BigWig file
        bw = pyBigWig.open(bw_filename, "w")

        # Add header - use end of genomic window as chromosome size
        bw.addHeader([(chrom, end)])

        # Add entries
        num_positions = len(track_data)
        starts = np.arange(pred_start, pred_start + num_positions, dtype=np.int64)
        ends = starts + 1
        values = track_data.tolist()

        bw.addEntries(
            chroms=[chrom] * len(starts),
            starts=starts.tolist(),
            ends=ends.tolist(),
            values=values,
        )

        bw.close()
        created_files.append(bw_filename)

    # Create zip file
    zip_path = os.path.join(tmpdir, f"ntv3_tracks_{uuid.uuid4().hex}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for bw_file in created_files:
            zipf.write(bw_file, os.path.basename(bw_file))

    # Clean up individual BigWig files
    for bw_file in created_files:
        try:
            os.remove(bw_file)
        except Exception:
            pass
    try:
        os.rmdir(output_dir)
    except Exception:
        pass

    return zip_path

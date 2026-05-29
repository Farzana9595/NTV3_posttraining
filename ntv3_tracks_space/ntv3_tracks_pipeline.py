from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.pipelines import Pipeline

try:
    import requests
except Exception:
    requests = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# ---------------------------------------------------------------------
# Assembly <-> species mapping
# ---------------------------------------------------------------------
ASSEMBLY_TO_SPECIES = {
    "hg38": "human",
    "mm10": "mouse",
    "dm6": "drosophila_melanogaster",
    "TAIR10": "arabidopsis_thaliana",
    "Zm-B73-REFERENCE-NAM-5.0": "zea_mays",
    "IRGSP-1.0": "oryza_sativa",
    "Glycine_max_v2.1": "glycine_max",
    "IWGSC": "triticum_aestivum",
    "Gossypium_hirsutum_v2.1": "gossypium_hirsutum",
    "AmpOce1": "amphiprion_ocellaris",
    "Bison_UMD1": "bison_bison_bison",
    "ChiLan1": "chinchilla_lanigera",
    "Felis_catus_9": "felis_catus",
    "GRCz11": "danio_rerio",
    "KH": "ciona_intestinalis",
    "Mnem_1": "macaca_nemestrina",
    "ROS_Cfam_1": "canis_lupus_familiaris",
    "SCA1": "serinus_canaria",
    "TETRAODON8": "tetraodon_nigroviridis",
    "WBcel235": "caenorhabditis_elegans",
    "bGalGal1": "gallus_gallus",
    "fSalTru1": "salmo_trutta",
    "gorGor4": "gorilla_gorilla",
    "mRatBN7": "rattus_norvegicus",
}
SPECIES_TO_ASSEMBLY = {v: k for k, v in ASSEMBLY_TO_SPECIES.items()}

# ---------------------------------------------------------------------
# Species that support coordinate-based sequence fetching
# ---------------------------------------------------------------------
# List of species that can fetch DNA sequences from genomic coordinates via API.
# Species not in this list can still be used but require direct DNA sequence input.
SPECIES_WITH_COORDINATE_SUPPORT = {
    "human",  # hg38 - UCSC API
    "mouse",  # mm10 - UCSC API
    "drosophila_melanogaster",  # dm6 - UCSC API
    "arabidopsis_thaliana",  # TAIR10 - UCSC hub API
    "gorilla_gorilla",  # gorGor4 - UCSC API
    # Add more species as API URLs are configured
}

# ---------------------------------------------------------------------
# Assembly -> API URL template mapping
# ---------------------------------------------------------------------
# Default API URL template (UCSC format) that works for most species
DEFAULT_API_URL_TEMPLATE = "https://api.genome.ucsc.edu/getData/sequence?genome={assembly};chrom={chrom};start={start};end={end}"  # noqa: E501

# for species with different format, add the assembly name to the mapping
# The template should use {chrom}, {start}, and {end} as placeholders.
ASSEMBLY_TO_API_URL_TEMPLATE = {
    # Arabidopsis thaliana (TAIR10) - uses hub URL format
    "TAIR10": "https://api.genome.ucsc.edu/getData/sequence?hubUrl=http://genome.ucsc.edu/goldenPath/help/examples/hubExamples/hubAssembly/plantAraTha1/hub.txt;genome=araTha1;chrom={chrom};start={start};end={end}",  # noqa: E501
}


# BED element to color mapping (shared between pipeline and app)
BED_ELEMENT_COLORS = {
    "protein coding gene": "#E74C3C",  # Red
    "lncRNA": "#2ECC71",  # Green
    "exon": "#9B59B6",  # Purple
    "intron": "#F39C12",  # Orange
    "splice_donor": "#1ABC9C",  # Teal
    "splice_acceptor": "#E67E22",  # Dark orange
    "CTCF-bound": "#3498DB",  # Light blue
    "polyA_signal": "#95A5A6",  # Gray
    "enhancer Tissue specific": "#D35400",  # Dark red
    "enhancer Tissue invariant": "#16A085",  # Dark teal
    "promoter Tissue specific": "#C0392B",  # Dark red 2
    "promoter Tissue invariant": "#27AE60",  # Dark green
    "5UTR+": "#8E44AD",  # Dark purple
    "5UTR-": "#D68910",  # Dark orange 2
    "3UTR+": "#138D75",  # Dark teal 2
    "3UTR-": "#2874A6",  # Dark blue
    "skipped exon": "#7D3C98",  # Purple 2
    "always on exon": "#A93226",  # Red 2
    "start codon": "#196F3D",  # Green 2
    "stop codon": "#B9770E",  # Brown
    "ORF": "#1F618D",  # Blue 2
}


def _filter_bed_elements_by_species(
    bed_element_names: list[str], species: str
) -> list[str]:
    """
    Filter BED element names based on species-specific training data availability.
    
    Rules:
    - Human: all tracks
    - Mouse: only polyA_signal
    - Other species: everything except promoter, enhancer, ctcf, lncrna
    
    Parameters
    ----------
    bed_element_names : list[str]
        Full list of BED element names from the model config
    species : str
        Species name (e.g., "human", "mouse", "drosophila_melanogaster")
    
    Returns
    -------
    list[str]
        Filtered list of BED element names available for this species
    """
    if not bed_element_names:
        return []
    
    # Elements to exclude for "other species" (everything except human and mouse)
    excluded_for_other_species = {
        "promoter Tissue specific",
        "promoter Tissue invariant",
        "enhancer Tissue specific",
        "enhancer Tissue invariant",
        "CTCF-bound",
        "lncRNA",
    }
    
    # Normalize element names (handle both with/without underscores/spaces)
    normalized_excluded = set()
    for elem in excluded_for_other_species:
        normalized_excluded.add(elem)
        normalized_excluded.add(elem.replace(" ", "_"))
    
    if species == "human":
        # Human: all tracks
        return list(bed_element_names)
    else:
        # Other species: everything except promoter, enhancer, ctcf, lncrna
        # Normalize element names for comparison (handle spaces, underscores, case)
        normalized_bed_names = {
            elem.lower().replace("_", " "): elem
            for elem in bed_element_names
        }
        normalized_excluded_lower = {
            elem.lower().replace("_", " ")
            for elem in excluded_for_other_species
        }
        
        # Also check for keywords in element names
        excluded_keywords = ["promoter", "enhancer", "ctcf", "lnc"]
        
        filtered_normalized = [
            norm_name
            for norm_name, orig_elem in normalized_bed_names.items()
            if norm_name not in normalized_excluded_lower
            and not any(keyword in norm_name for keyword in excluded_keywords)
        ]
        
        # Return original element names (preserving original format)
        return [
            normalized_bed_names[norm_name] for norm_name in filtered_normalized
        ]


def _sanitize_dna(seq: str) -> str:
    seq = seq.upper()
    return "".join(ch if ch in ("A", "C", "G", "T", "N") else "N" for ch in seq)


def _get_dna_sequence(assembly: str, chrom: str, start: int, end: int) -> str:
    """
    Fetch DNA sequence from API based on assembly, chromosome, and coordinates.

    Uses ASSEMBLY_TO_API_URL_TEMPLATE to determine the API URL format for each assembly.
    Falls back to DEFAULT_API_URL_TEMPLATE if assembly is not in the mapping.
    """
    if requests is None:
        raise ImportError(
            "requests is required for genome download. "
            "Install with: pip install requests"
        )

    # Get API URL template for this assembly, or use default
    url_template = ASSEMBLY_TO_API_URL_TEMPLATE.get(assembly, DEFAULT_API_URL_TEMPLATE)

    # Format the URL with the provided parameters
    url = url_template.format(assembly=assembly, chrom=chrom, start=start, end=end)

    seq = requests.get(url).json()["dna"].upper()
    return seq


def _pick_device(device: str | int | torch.device) -> torch.device:
    # Handle torch.device objects
    if isinstance(device, torch.device):
        return device

    # Handle integer device IDs (transformers pipeline convention)
    if isinstance(device, int):
        if device == -1:
            return torch.device("cpu")
        elif device >= 0:
            if torch.cuda.is_available():
                return torch.device(f"cuda:{device}")
            else:
                return torch.device("cpu")
        else:
            raise ValueError(f"Invalid device integer: {device}")

    # Handle string device names
    if isinstance(device, str):
        d = device.lower()
        if d == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        if d in ("cuda", "cpu", "mps"):
            return torch.device(d)
        raise ValueError(
            "device must be one of: 'auto', 'cpu', 'cuda', 'mps', or an integer"
        )

    raise ValueError(
        f"device must be a string, integer, or torch.device, got {type(device)}"
    )


def _softmax_last(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=-1, keepdims=True)


def _plot_tracks_fillbetween(
    tracks: dict[str, np.ndarray],
    chrom: str | None,
    start: int,
    end: int,
    assembly: str | None,
    height: float = 1.0,
    figsize_x: float = 20.0,
):
    if plt is None:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        )

    n = len(tracks)
    if n == 0:
        raise ValueError("No tracks to plot.")

    fig, axes = plt.subplots(n, 1, figsize=(figsize_x, height * n), sharex=True)
    if n == 1:
        axes = [axes]

    any_track = next(iter(tracks.values()))
    x = np.linspace(start, end, num=len(any_track), endpoint=False)

    # Define color schemes
    # BigWig tracks: use blue/gray tones
    bigwig_color = "#4A90E2"  # Blue

    for ax, (title, y) in zip(axes, tracks.items()):
        # Determine color based on track type
        if title in BED_ELEMENT_COLORS:
            color = BED_ELEMENT_COLORS[title]
        else:
            color = bigwig_color

        ax.fill_between(x, y, color=color, alpha=0.3, linewidth=0)
        ax.plot(x, y, color=color, linewidth=0.8)
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(alpha=0.2)
        ax.set_yticks([])
        # minimal "despine"
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    label = f"{chrom}:{start}-{end}" if chrom is not None else f"{start}-{end}"
    if assembly is not None:
        label += f" ({assembly})"
    axes[-1].set_xlabel(label)

    plt.tight_layout()
    return fig, axes


@dataclass
class NTv3TracksOutput:
    bigwig_tracks_logits: np.ndarray  # (L_pred, T)
    bed_tracks_logits: np.ndarray  # (L_pred, E, C)
    mlm_logits: np.ndarray
    chrom: str | None = None
    start: int | None = None
    end: int | None = None
    species: str | None = None
    assembly: str | None = None
    bigwig_track_names: list[str] | None = (
        None  # from cfg.bigwigs_per_species[species]
    )
    bed_element_names: list[str] | None = None
    window_len: int | None = None
    pred_start: int | None = None
    pred_end: int | None = None


class NTv3TracksPipeline(Pipeline):
    def __init__(
        self,
        model: str | torch.nn.Module,
        tokenizer: str | Any | None = None,
        trust_remote_code: bool = True,
        token: str | None = None,
        default_species: str = "human",
        genome_cache_dir: str | Path = "~/.cache/ntv3/genomes",
        device: str = "auto",
        mps_force_cpu: bool = True,
        mps_force_cpu_length: int = 16384,
        verbose: bool = True,
        # Your notebook uses these constants for "middle 37.5%" prediction span
        pred_center_fraction: float = 0.375,
        pred_center_offset_fraction: float = 0.3125,
        **kwargs: Any,
    ):
        self.model_id = model if isinstance(model, str) else None
        self.default_species = default_species
        self.genome_cache_dir = Path(genome_cache_dir)
        self.mps_force_cpu = bool(mps_force_cpu)
        self.mps_force_cpu_length = int(mps_force_cpu_length)
        self.verbose = bool(verbose)
        self.pred_center_fraction = float(pred_center_fraction)
        self.pred_center_offset_fraction = float(pred_center_offset_fraction)

        if isinstance(model, str):
            self.config = AutoConfig.from_pretrained(
                model, trust_remote_code=trust_remote_code, token=token
            )
            self.model = AutoModel.from_pretrained(
                model, trust_remote_code=trust_remote_code, token=token
            )
        else:
            self.model = model
            self.config = getattr(model, "config", None)

        if tokenizer is None:
            if not self.model_id:
                raise ValueError(
                    "If passing a model module, pass tokenizer explicitly."
                )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, trust_remote_code=trust_remote_code, token=token,
            )
        elif isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer, trust_remote_code=trust_remote_code, token=token
            )
        else:
            self.tokenizer = tokenizer

        # Extract model_id from config if not already set
        # (following ntv3_gff_pipeline.py pattern)
        if self.model_id is None and self.config is not None:
            self.model_id = getattr(self.config, "_name_or_path", None) or getattr(
                self.config, "name_or_path", None
            )

        # bed names (your notebooks refer to bed_element_names)
        self.bed_element_names = getattr(
            self.config, "bed_elements_names", None
        ) or getattr(self.config, "bed_element_names", None)

        self._target_device = _pick_device(device)
        self.model.to(self._target_device)
        self.model.eval()

        super().__init__(
            model=self.model, tokenizer=self.tokenizer, device=-1, **kwargs
        )

    def available_bigwig_track_names(self, species: str | None = None) -> list[str]:
        """
        Return BigWig track IDs for the assembly corresponding to `species`.
        No model forward pass.
        """
        if species not in self.config.bigwigs_per_species:
            raise ValueError(
                f"Species {species} not found in checkpoint config. "
                f"Available: {list(self.config.bigwigs_per_species.keys())}"
            )

        return list(self.config.bigwigs_per_species[species])

    def available_bed_element_names(self, species: str | None = None) -> list[str]:
        """
        Return BED element names available in this checkpoint for the given species.
        Filters elements based on species-specific training data availability.
        
        Parameters
        ----------
        species : str | None
            Species name (e.g., "human", "mouse"). If None, returns all elements
            without filtering (for backward compatibility).
        
        Returns
        -------
        list[str]
            Filtered list of BED element names available for this species
        """
        all_elements = list(self.bed_element_names or [])
        if species is None:
            return all_elements
        return _filter_bed_elements_by_species(all_elements, species)

    def _sanitize_parameters(self, **kwargs):
        return {}, {}, {}

    def _get_model_device(self) -> torch.device:  # noqa: CCE001
        return next(self.model.parameters()).device

    def _resolve_species_and_assembly(self, inputs: dict[str, Any]) -> tuple[str, str]:
        species = inputs.get("species", self.default_species)
        if species not in SPECIES_TO_ASSEMBLY:
            supported = sorted(SPECIES_TO_ASSEMBLY.keys())
            raise ValueError(
                f"Unsupported species='{species}'. " f"Supported species: {supported}"
            )
        assembly = SPECIES_TO_ASSEMBLY[species]

        cfg_species = list(self.config.bigwigs_per_species.keys())
        if species not in cfg_species:
            raise ValueError(
                f"Species '{species}' is not available in this checkpoint. "
                f"Available species: {cfg_species}"
            )
        return species, assembly

    def _maybe_force_cpu_for_mps_long(  # noqa: CCE001
        self, input_ids_cpu: torch.Tensor
    ) -> torch.device:
        dev = self._get_model_device()
        if self.mps_force_cpu and dev.type == "mps":
            seq_len = int(input_ids_cpu.shape[-1])
            if seq_len >= self.mps_force_cpu_length:
                if self.verbose:
                    print(
                        f"[NTv3TracksPipeline] MPS detected and input is long "
                        f"(tokens={seq_len}). Switching model + inputs to CPU "
                        "for this run."
                    )
                self.model.to("cpu")
                self.model.eval()
                return torch.device("cpu")
        return dev

    def preprocess(self, inputs: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        species, assembly = self._resolve_species_and_assembly(inputs)

        # Resolve sequence
        if "seq" in inputs and inputs["seq"] is not None:
            seq = _sanitize_dna(inputs["seq"])
            chrom = None
            start = 0
            end = len(seq)
            window_len = len(seq)
        else:
            chrom = inputs["chrom"]
            start = int(inputs["start"])
            end = int(inputs["end"])
            window_len = end - start
            seq = _get_dna_sequence(assembly, chrom, start, end)
            seq = _sanitize_dna(seq)

        # Tokenize with padding
        batch = self.tokenizer(
            [seq],
            add_special_tokens=False,
            padding=True,
            pad_to_multiple_of=128,
            return_tensors="pt",
        )
        input_ids_cpu = batch["input_ids"]

        # MPS-long fallback decision
        device = self._maybe_force_cpu_for_mps_long(input_ids_cpu)

        # Move inputs
        input_ids = input_ids_cpu.to(device)
        # Species tokenization - match batch size
        batch_size = input_ids.shape[0]
        species_ids = self.model.encode_species([species] * batch_size)
        species_ids_tensor = species_ids.to(device)

        # Prediction interval (not used for slicing logits, just x-axis)
        pred_start = start + int(window_len * self.pred_center_offset_fraction)
        pred_end = pred_start + int(window_len * self.pred_center_fraction)

        # ✅ The source of truth for track IDs/names (your note)
        bigwig_track_names = list(self.config.bigwigs_per_species[species])

        return {
            "input_ids": input_ids,
            "species_ids": species_ids_tensor,
            "meta": {
                "chrom": chrom,
                "start": start,
                "end": end,
                "species": species,
                "assembly": assembly,
                "window_len": window_len,
                "pred_start": pred_start,
                "pred_end": pred_end,
                "bigwig_track_names": bigwig_track_names,
            },
        }

    # prevent Pipeline from moving tensors to its own device
    def forward(self, model_inputs, **forward_params):
        return self._forward(model_inputs, **forward_params)

    def postprocess(
        self, model_outputs: dict[str, Any], **kwargs: Any
    ) -> NTv3TracksOutput:
        # Extract model_output and meta from the dict returned by _forward
        if isinstance(model_outputs, dict) and "model_output" in model_outputs:
            model_out = model_outputs["model_output"]
            meta = model_outputs.get("meta", {})
        else:
            # Fallback for direct ModelOutput (shouldn't happen with current code)
            model_out = model_outputs
            meta = {}

        def to_np(x):
            return x.detach().float().cpu().numpy()

        # Access model output - ModelOutput objects support both dict and attribute access
        bigwig_np = to_np(model_out["bigwig_tracks_logits"])
        bed_np = to_np(model_out["bed_tracks_logits"])
        mlm_np = to_np(model_out["logits"])

        # Normalize shapes to remove batch/(optional assembly) dims
        if bigwig_np.ndim == 3:
            bigwig_np = bigwig_np[0]  # (L, T)
        elif bigwig_np.ndim == 4:
            bigwig_np = bigwig_np[0, 0]  # (L, T) if (B, A, L, T)
        else:
            raise ValueError(f"Unexpected bigwig_tracks_logits ndim: {bigwig_np.ndim}")

        if bed_np.ndim == 4:
            bed_np = bed_np[0]  # (L, E, C)
        elif bed_np.ndim == 5:
            bed_np = bed_np[0, 0]  # (L, E, C) if (B, A, L, E, C)
        else:
            raise ValueError(f"Unexpected bed_tracks_logits ndim: {bed_np.ndim}")

        if mlm_np.ndim == 3:
            mlm_np = mlm_np[0]

        # Filter BED elements based on species
        species = meta.get("species")
        all_bed_element_names = self.bed_element_names or []
        if species and all_bed_element_names:
            filtered_bed_element_names = _filter_bed_elements_by_species(
                all_bed_element_names, species
            )
            # Filter bed_tracks_logits to only include elements available for this species
            if filtered_bed_element_names != all_bed_element_names:
                # Create mapping from filtered element names to original indices
                element_indices = [
                    all_bed_element_names.index(elem)
                    for elem in filtered_bed_element_names
                    if elem in all_bed_element_names
                ]
                if element_indices:
                    # bed_np shape is (L, E, C) where E is number of elements
                    bed_np = bed_np[:, element_indices, :]
                    # Update filtered list to only include elements that were found
                    filtered_bed_element_names = [
                        elem
                        for elem in filtered_bed_element_names
                        if elem in all_bed_element_names
                    ]
        else:
            filtered_bed_element_names = all_bed_element_names

        return NTv3TracksOutput(
            bigwig_tracks_logits=bigwig_np,
            bed_tracks_logits=bed_np,
            mlm_logits=mlm_np,
            chrom=meta.get("chrom"),
            start=meta.get("start"),
            end=meta.get("end"),
            species=meta.get("species"),
            assembly=meta.get("assembly"),
            bigwig_track_names=meta.get("bigwig_track_names"),
            bed_element_names=filtered_bed_element_names,
            window_len=meta.get("window_len"),
            pred_start=meta.get("pred_start"),
            pred_end=meta.get("pred_end"),
        )

    def _forward(self, model_inputs: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        meta = model_inputs.pop("meta")
        if self.verbose:
            print(f"Running on device: {self._get_model_device()}")
        with torch.no_grad():
            out = self.model(
                input_ids=model_inputs["input_ids"],
                species_ids=model_inputs["species_ids"],
            )
        # Return a dict containing the model output and meta separately
        # since ModelOutput objects are immutable
        return {"model_output": out, "meta": meta}

    def __call__(
        self,
        inputs,
        *args,
        plot: bool = False,
        tracks_to_plot: dict[str, str] | None = None,  # title -> track_id (ENCSR...)
        elements_to_plot: list[str] | None = None,  # element names
        plot_height: float = 1.0,
        plot_figsize_x: float = 20.0,
        **kwargs,
    ):
        """
        One-step call that can optionally plot and always returns NTv3TracksOutput.
        """
        out: NTv3TracksOutput = super().__call__(inputs, *args, **kwargs)

        if plot:
            if out.bigwig_track_names is None:
                raise ValueError(
                    "bigwig_track_names missing; expected "
                    "cfg.bigwigs_per_species[species]."
                )
            if out.bed_element_names is None:
                raise ValueError("bed element names missing from config.")
            tracks_to_plot = tracks_to_plot or {}
            elements_to_plot = elements_to_plot or []

            bigwig_names = out.bigwig_track_names
            bed_element_names = out.bed_element_names

            # Validate
            missing_tracks = [
                tid for tid in tracks_to_plot.values() if tid not in bigwig_names
            ]
            if missing_tracks:
                raise ValueError(
                    f"The following tracks are not available in "
                    f"bigwig_names: {missing_tracks}\n"
                    f"First 50 available: {bigwig_names[:50]}"
                    f"{'...' if len(bigwig_names) > 50 else ''}"
                )

            missing_elements = [
                e for e in elements_to_plot if e not in bed_element_names
            ]
            if missing_elements:
                first_50 = bed_element_names[:50]
                ellipsis = "..." if len(bed_element_names) > 50 else ""
                raise ValueError(
                    f"The following elements are not available in "
                    f"bed_element_names: {missing_elements}\n"
                    f"First 50 available: {first_50}{ellipsis}"
                )

            # Build bigwig tracks dict (title -> y)
            bigwig_tracks: dict[str, np.ndarray] = {}
            bigwig = out.bigwig_tracks_logits  # (L_pred, T)
            for title, track_id in tracks_to_plot.items():
                track_idx = bigwig_names.index(track_id)
                bigwig_tracks[title] = bigwig[:, track_idx]

            # Bed positive class probabilities (title -> y)
            bed_probs: dict[str, np.ndarray] = {}
            probs = _softmax_last(out.bed_tracks_logits)  # (L_pred, E, C)
            for element_name in elements_to_plot:
                element_idx = bed_element_names.index(element_name)
                bed_probs[element_name] = probs[:, element_idx, 1]

            all_tracks = {**bigwig_tracks, **bed_probs}

            plot_start = int(out.pred_start or 0)
            plot_end = int(
                out.pred_end or (plot_start + len(next(iter(all_tracks.values()))))
            )

            _plot_tracks_fillbetween(
                all_tracks,
                chrom=out.chrom,
                start=plot_start,
                end=plot_end,
                assembly=out.assembly,
                height=plot_height,
                figsize_x=plot_figsize_x,
            )

        return out


def load_ntv3_tracks_pipeline(
    model: str,
    device: str = "auto",
    **pipeline_kwargs: Any,
):
    """
    Convenience helper to build an NTv3TracksPipeline for any NTv3 checkpoint.

    Parameters
    ----------
    model:
        Checkpoint id, e.g. "InstaDeepAI/NTv3_100M", "InstaDeepAI/NTv3_650M", ...
    device:
        "auto", "cpu", "cuda", "mps"
    pipeline_kwargs:
        Extra kwargs passed to NTv3TracksPipeline
        (default_species, genome_cache_dir, etc.).
    """
    pipe = NTv3TracksPipeline(
        model=model,
        trust_remote_code=True,
        device=device,
        **pipeline_kwargs,
    )
    return pipe

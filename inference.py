"""
ISLES 2026 — Ensemble inference for Grand Challenge.

Loads three nnUNet models (ResEnc U-Net, PrimusV3S, MedNext) and averages
their predicted probabilities for the final segmentation.

Model weights are expected at /opt/ml/model/<model_name>/ (Grand Challenge
extracts the uploaded tarball there). For local testing the model/ directory
is bind-mounted instead.

Interfaces:
  interf0:
    Inputs:
      - /input/images/t1-brain-mri
      - /input/stroke-metadata.json
    Outputs:
      - /output/images/stroke-lesion-segmentation  (binary .mha)
      - /output/images/lesion-probability-map      (float32 .mha)
"""

import glob
import json
from pathlib import Path

import numpy
import SimpleITK
import torch

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")

MODEL_DIR = Path("/opt/ml/model")

MODEL_CONFIGS = [
    {
        "name": "resenc",
        "dir": MODEL_DIR / "resenc",
        "folds": None,  # auto-detect fold_0 … fold_4
        "checkpoint": "checkpoint_best.pth",
    },
    {
        "name": "primus",
        "dir": MODEL_DIR / "primus",
        "folds": None,  # auto-detect fold_0 … fold_4
        "checkpoint": "checkpoint_best.pth",
    },
    {
        "name": "mednext",
        "dir": MODEL_DIR / "mednext",
        "folds": ["all"],  # only fold_all exists
        "checkpoint": "checkpoint_best.pth",
    },
]


def _show_torch_cuda_info():
    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: {(current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


def init_model():
    """Load all three nnUNet predictors at server startup.

    Returns a list of (name, nnUNetPredictor) tuples.
    """
    _show_torch_cuda_info()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    predictors = []
    for cfg in MODEL_CONFIGS:
        print(f"\nLoading model: {cfg['name']} from {cfg['dir']}")
        p = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=device,
        )
        p.initialize_from_trained_model_folder(
            model_training_output_dir=str(cfg["dir"]),
            use_folds=cfg["folds"],
            checkpoint_name=cfg["checkpoint"],
        )
        predictors.append((cfg["name"], p))
        print(f"  -> loaded {cfg['name']} "
              f"(trainer={p.trainer_name}, "
              f"folds={[f for f in (cfg['folds'] or [])] or 'auto'}, "
              f"mirrors={p.allowed_mirroring_axes})")

    print(f"\nEnsemble ready: {len(predictors)} models loaded")
    return predictors


def run(model):
    """Called once per /invoke. Reads T1w input, runs ensemble, writes output."""
    interface_key = get_interface_key()

    handler = {
        ("stroke-metadata", "t1-brain-mri"): interf0_handler,
    }[interface_key]

    return handler(model)


def interf0_handler(predictors):
    # ---- Load input image ------------------------------------------------
    t1_image, t1_data = load_image_file_as_array_and_image(
        location=INPUT_PATH / "images/t1-brain-mri",
    )

    # Load metadata (logged but not used for inference)
    input_stroke_metadata = load_json_file(
        location=INPUT_PATH / "stroke-metadata.json",
    )
    print("=+=" * 10)
    print("Loaded Stroke Metadata:")
    print(json.dumps(input_stroke_metadata, indent=2))
    print("=+=" * 10)

    # ---- Prepare input for nnUNet ----------------------------------------
    # SimpleITK GetSpacing returns (x, y, z); numpy array from
    # GetArrayFromImage is (z, y, x).  Reverse to match array axes.
    spacing = list(reversed(t1_image.GetSpacing()))
    # nnUNet expects (C, X, Y, Z) — add channel dimension
    image_np = t1_data.astype(numpy.float32)[numpy.newaxis]

    # ---- Run each model and collect probability maps ----------------------
    all_probs = []

    for name, predictor in predictors:
        print(f"\n{'='*20} Predicting with {name} {'='*20}")
        seg, probs = predictor.predict_single_npy_array(
            input_image=image_np,
            image_properties={"spacing": spacing},
            save_or_return_probabilities=True,
        )
        # probs shape: (num_classes, D, H, W) — already resampled to original shape
        all_probs.append(probs)
        print(f"  {name} done — probs shape {probs.shape}, "
              f"range [{probs.min():.4f}, {probs.max():.4f}]")

        # Free GPU memory between models
        torch.cuda.empty_cache()

    # ---- Ensemble: average probabilities ---------------------------------
    avg_probs = numpy.mean(all_probs, axis=0).astype(numpy.float32)

    # Binary segmentation: argmax over class dimension
    binary_segmentation_mask = numpy.argmax(avg_probs, axis=0).astype(numpy.uint8)

    # Lesion probability map: class-1 probabilities
    probability_map_data = avg_probs[1]

    # Log summary statistics
    n_voxels = int(binary_segmentation_mask.sum())
    print(f"\nEnsemble result: {n_voxels} lesion voxels "
          f"({n_voxels / binary_segmentation_mask.size * 100:.2f}% of volume)")
    print(f"Lesion prob range: [{probability_map_data.min():.4f}, "
          f"{probability_map_data.max():.4f}]")

    # ---- Save outputs (preserve input spatial metadata) -------------------
    write_array_as_image_file(
        location=OUTPUT_PATH / "images/stroke-lesion-segmentation",
        array=binary_segmentation_mask,
        reference_image=t1_image,
    )

    write_array_as_image_file(
        location=OUTPUT_PATH / "images/lesion-probability-map",
        array=probability_map_data,
        reference_image=t1_image,
    )

    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_interface_key():
    inputs = load_json_file(location=INPUT_PATH / "inputs.json")
    socket_slugs = [sv["socket"]["slug"] for sv in inputs]
    return tuple(sorted(socket_slugs))


def load_json_file(*, location):
    with open(location) as f:
        return json.loads(f.read())


def load_image_file_as_array_and_image(*, location):
    input_files = (
        glob.glob(str(location / "*.mha"))
        + glob.glob(str(location / "*.nii.gz"))
        + glob.glob(str(location / "*.nii"))
    )
    if not input_files:
        raise FileNotFoundError(f"No valid image file found in {location}")

    image = SimpleITK.ReadImage(input_files[0])
    return image, SimpleITK.GetArrayFromImage(image)


def write_array_as_image_file(*, location, array, reference_image=None):
    location.mkdir(parents=True, exist_ok=True)

    image = SimpleITK.GetImageFromArray(array)

    if reference_image is not None:
        image.SetSpacing(reference_image.GetSpacing())
        image.SetOrigin(reference_image.GetOrigin())
        image.SetDirection(reference_image.GetDirection())

    SimpleITK.WriteImage(
        image,
        location / "output.mha",
        useCompression=True,
    )


if __name__ == "__main__":
    raise SystemExit(run(model=init_model()))

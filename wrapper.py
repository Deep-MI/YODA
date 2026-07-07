import argparse
import os
import json
import shutil

from omegaconf import OmegaConf
import tempfile

import numpy as np

from lib.utils.etc import mini_yoda, the_force
from scripts.preprocessing.conform import conform

# usage:
# python predict/wrapper.py -r </path/to/run> -c </path/to/specific/ckpt.pth> \
#     -o </path/to/save/predictions or .../prediction/filename.nii.gz> \
#     <input nii files>  --config </path/to/config.json (optional)> \
#     --mask </path/to/mask.nii.gz (optional)>

# NOTE: channel order matters, it hast to match the respective run config

# assumption:
# run directory structure is as follows:
# ├── path/to/run       # set via -r/--run
# │   ├── config.yml    # has to exists here
# │   └── ckpt/last.pth # configurable via -c/--checkpoint

parser = argparse.ArgumentParser(
    description="Wrapper for YODA (translation)/ YADO (denoising) prediction"
)

parser.add_argument(
    "-r",
    "--run",
    type=str,
    required=True,
    help="Path to the run directory containing the config and checkpoint.",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default="ckpt/last.pth",
    help="Path to the checkpoint file within --run directory (e.g. ckpt/last.pth) or directly to a checkpoint file.",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    required=True,
    help="Path to save the predictions (either a directory or a specific file).",
)
parser.add_argument(
    "input_files",
    nargs="+",
    help="Input NIfTI files for prediction. Note that the order matters (YODA: alphabetical (e.g. T1w,T2w), YADO: denoised sequence, then guidance alphabetically (e.g. T1w denoising could be input T1w,FLAIR,T2w)).",
)
parser.add_argument(
    "--mask",
    "-m",
    type=str,
    default=None,
    help="Path to a brain mask NIfTI file to apply to the predictions (optional). If no mask is provided, a dummy center mask is created to restrict the ROI to the model's img_size.",
)
parser.add_argument(
    "--config",
    type=str,
    default=None,
    help="Additional YAML to overwrite model defaults.",
)
parser.add_argument(
    "--axial_only",
    action="store_true",
    help="If set, only the axial slices will be predicted to speed prediction up.",
)
parser.add_argument("-f", "--force", action="store_true", help="Use the force, Luke.")
parser.add_argument(
    "--skull_strip",
    "-s",
    action="store_true",
    help="Apply brain mask to the denoiser input.",
)
parser.add_argument(
    "--crop", "-c", action="store_true", help="Crop to max input size of the model"
)
args = parser.parse_args()

if args.force:
    print("YODA predicting your images is")
    print("Patience you must have, my young Padawan!")
    print("\n" + mini_yoda + "\n")

config = OmegaConf.load(args.run + "/config.yml")
if args.config:
    config = OmegaConf.merge(config, OmegaConf.load(args.config))

input_seqs = config.data.guidance_sequences
assert len(args.input_files) == len(
    input_seqs
), f"Number of input files ({len(args.input_files)}) must match number of guidance sequences in config ({len(input_seqs)})."

if not args.output.endswith(".nii.gz"):
    args.output = os.path.join(args.output, "pred.nii.gz")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

# conform
tmp = tempfile.mkdtemp()


def _copy_outputs(subject_dir: str, output_path: str) -> str:
    """Copy prediction NIfTI(s) from a subject folder to `output_path`.

    If output_path ends with .nii.gz: copy one best-effort file there.
    Otherwise: treat it as a directory and copy all pred*.nii.gz to it.
    """

    if not os.path.isdir(subject_dir):
        raise FileNotFoundError(f"Expected subject output dir not found: {subject_dir}")

    preds = sorted(
        f
        for f in os.listdir(subject_dir)
        if f.endswith(".nii.gz") and (f == "pred.nii.gz" or f.startswith("pred_"))
    )
    if not preds:
        # fall back to any pred*.nii.gz
        preds = sorted(
            f
            for f in os.listdir(subject_dir)
            if f.endswith(".nii.gz") and f.startswith("pred")
        )

    if not preds:
        raise FileNotFoundError(
            f"No prediction NIfTI found in {subject_dir}. Found: {sorted(os.listdir(subject_dir))}"
        )

    if output_path.endswith(".nii.gz"):
        preferred = "pred.nii.gz" if "pred.nii.gz" in preds else preds[0]
        shutil.copy2(os.path.join(subject_dir, preferred), output_path)
        return output_path

    os.makedirs(output_path, exist_ok=True)
    primary = ""
    for f in preds:
        dst = os.path.join(output_path, f)
        shutil.copy2(os.path.join(subject_dir, f), dst)
        if not primary:
            primary = dst
    return primary


def _create_dummy_center_mask(
    reference_nii: str, out_path: str, spatial_size_dhw
) -> str:
    """Create a dummy mask with ones in the central region, else zeros.

    spatial_size_dhw is (D, H, W) foreground extent (WITHOUT margin).

    Behavior:
      - H/W are centered.
      - D (superior/inferior) is placed at the *uppermost/superior* end of the volume.
        Depending on orientation, this may correspond to either the start (0) or end (-1)
        of the depth axis. We place it at the high-index end to match common datasets where
        the head is at the max index.
    """

    import nibabel as nib

    ref = nib.load(reference_nii)
    data_shape = ref.shape
    if len(data_shape) != 3:
        if len(data_shape) == 4 and data_shape[-1] == 1:
            data_shape = data_shape[:3]
        else:
            raise ValueError(
                f"Expected 3D NIfTI reference for dummy mask, got shape {ref.shape}"
            )

    x, y, z = data_shape
    d_t, h_t, w_t = (
        int(spatial_size_dhw[0]),
        int(spatial_size_dhw[1]),
        int(spatial_size_dhw[2]),
    )

    # H/W centered
    start_h = max((x - h_t) // 2, 0)
    start_w = max((y - w_t) // 2, 0)

    # D placed at high-index end (keep head, drop neck)
    d_t = min(d_t, z)
    start_d = max(z - d_t - 10, 0)

    end_h = min(start_h + h_t, x)
    end_w = min(start_w + w_t, y)
    end_d = z - 10

    mask = np.zeros((x, y, z), dtype=np.uint8)
    mask[start_h:end_h, start_w:end_w, start_d:end_d] = 1

    out = nib.Nifti1Image(mask, ref.affine, ref.header)
    out.set_data_dtype(np.uint8)
    nib.save(out, out_path)
    return out_path


try:
    conformed_paths = []
    for i, input_file in enumerate(args.input_files):
        conformed_path = os.path.join(tmp, os.path.basename(input_file))
        conform(input_file, conformed_path, dry_run=False)
        conformed_paths.append(conformed_path)

    # create dummy MONAI Dataset JSON
    ds = {"subject_ID": "Dummy"}
    for seq, input_file in zip(input_seqs, args.input_files):
        ds[seq] = os.path.join(tmp, os.path.basename(input_file))

    if args.mask:
        ds["mask"] = args.mask
    else:
        img_size = getattr(config.data, "img_size", None)
        if img_size is None or len(img_size) < 2:
            raise ValueError(
                "No --mask provided and config.data.img_size is missing/invalid; can't build dummy mask."
            )

        # crop_to_brain_margin is 3D (D, H, W) in the SPL-oriented space.
        # For the dummy center mask, we assume/require an isotropic margin (same across axes),
        # and construct a *cube* ROI of side (cube_side - 2*margin) so that after CropForegroundd
        # adds +/- margin, the effective crop fits the model's expected img_size.
        margin = getattr(config.data, "crop_to_brain_margin", None)
        if margin is None:
            m = 0
        else:
            m_d, m_h, m_w = int(margin[0]), int(margin[1]), int(margin[2])
            m = max(m_d, m_h, m_w)

        cube_side = int(min(img_size[0], img_size[1]))
        fg_cube = max(cube_side - 2 * m, 1)

        dummy_mask_path = os.path.join(tmp, "dummy_center_mask.nii.gz")
        ds["mask"] = _create_dummy_center_mask(
            reference_nii=conformed_paths[0],
            out_path=dummy_mask_path,
            spatial_size_dhw=(fg_cube, fg_cube, fg_cube),
        )

    jason = {"training": [], "validation": [ds]}
    with open(os.path.join(tmp, "dataset.json"), "w") as f:
        json.dump(jason, f)

    config.data.dataset = os.path.join(tmp, "dataset.json")
    config.data.data_dir = ""
    # For inference, we don't require ground-truth targets.
    config.data.target_sequence = ""
    config.data.skull_strip = args.skull_strip

    # Keep mask-based ROI cropping enabled (dummy mask is provided if real mask is absent).

    # run prediction
    out_root = os.path.join(tmp, "out")
    os.makedirs(out_root, exist_ok=True)

    # user-facing output
    output_is_file = args.output.endswith(".nii.gz")
    output_dir = os.path.dirname(args.output) if output_is_file else args.output
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if "train_noise_sched" not in config.model.keys():
        from predict.reg_predict import main as reg_main

        reg_main(
            args.run,
            out_root,
            args.checkpoint,
            config.copy(),
            subject_indices=[0],
            check_if_exists=False,
            view_agg=(not args.axial_only),
            crop=args.crop,
        )

        # reg_predict saves: <out_root>/<subject_ID>/*.nii.gz
        _copy_outputs(os.path.join(out_root, "Dummy"), args.output)

    # legacy DM YODA, only supports Regression (it is all you need, anyways, except for augmentation)
    else:
        from predict.dm_predict import main as dm_main

        # hard-coded Regression sampling
        config.model.train_noise_sched["type"] = "DDIM"
        config.model.train_noise_sched["timestep_spacing"] = "linspace"
        config.model.num_inference_steps = 1

        regression = OmegaConf.load("configs/inference_schedulers/Regression.yml")
        config = OmegaConf.merge(config, regression)

        # axial
        dm_main(
            args.run,
            out_root,
            args.checkpoint,
            config.copy(),
            subject_indices=[0],
            crop=args.crop,
        )

        if args.axial_only:
            _copy_outputs(os.path.join(out_root, "Dummy"), args.output)
        else:
            # sagittal
            out_sag = os.path.join(tmp, "out_sag")
            os.makedirs(out_sag, exist_ok=True)
            config.data.slicing_direction = "sagittal"
            dm_main(
                args.run,
                out_sag,
                args.checkpoint,
                config.copy(),
                subject_indices=[0],
                crop=args.crop,
            )

            # coronal
            out_cor = os.path.join(tmp, "out_cor")
            os.makedirs(out_cor, exist_ok=True)
            config.data.slicing_direction = "coronal"
            dm_main(
                args.run,
                out_cor,
                args.checkpoint,
                config.copy(),
                subject_indices=[0],
                crop=args.crop,
            )

            import nibabel as nib

            # Find a prediction file name present in all 3 folders.
            subj_dirs = [
                os.path.join(out_root, "Dummy"),
                os.path.join(out_sag, "Dummy"),
                os.path.join(out_cor, "Dummy"),
            ]
            pred_sets = []
            for d in subj_dirs:
                pred_sets.append(
                    {
                        f
                        for f in os.listdir(d)
                        if f.endswith(".nii.gz")
                        and (f.startswith("pred") or f.startswith("pred_"))
                    }
                )
            common = sorted(set.intersection(*pred_sets))
            if not common:
                raise FileNotFoundError(
                    "Couldn't find a common prediction file to ensemble across views. "
                    f"Candidates: {[sorted(s) for s in pred_sets]}"
                )
            pred_name = "pred.nii.gz" if "pred.nii.gz" in common else common[0]

            imgs = []
            aff = None
            for d in subj_dirs:
                img = nib.load(os.path.join(d, pred_name))
                imgs.append(img.get_fdata())
                if aff is None:
                    aff = img.affine
            # RMS combine
            combined = np.sqrt(np.mean(np.array(imgs) ** 2, axis=0))
            out_img = nib.Nifti1Image(combined, aff)

            if args.output.endswith(".nii.gz"):
                nib.save(out_img, args.output)
            else:
                os.makedirs(args.output, exist_ok=True)
                nib.save(out_img, os.path.join(args.output, pred_name))

finally:
    shutil.rmtree(tmp, ignore_errors=True)

print(f"Prediction complete! Saved to {args.output}")
# citation
print(
    f"""\n
    If you use YODA in your research, please consider citing:
    Rassmann, S., Kügler, D., Ewert, C., & Reuter, M. (2026). 
    Regression is all you need for medical image translation. 
    IEEE Transactions on Medical Imaging
    """
)


if args.force:
    print("\n" + the_force + "\n")

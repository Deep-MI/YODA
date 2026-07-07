import warnings
import sys, os

sys.path.append(os.path.realpath(os.getcwd()))

import argparse
import torch
import matplotlib.pyplot as plt  # debug

from lib.datasets import get_datasets
from lib.inference import (
    save_output_volume,
    save_output_png,
    Sr3InferenceModel,
)
from lib.utils.run_utils import merge_cli_config
from lib.utils.etc import remove_module_in_state_dict, seed_from_subject_id
from lib.utils.monai_helper import parse_dtype, scheduler_factory
from lib.utils.etc import the_force, mini_yoda

device = "cuda"


def main(
    run_name,
    output_dir,
    ckpt_name,
    config,
    subject_indices,
    n_slices_per_sample=None,
    seed_offset=0,
    quantile_norm=False,
    tqdm_verbose=True,
    check_if_exists=False,
    crop=False,
):
    dtype = parse_dtype(config.trainer.dtype)
    if not "/" in ckpt_name:
        ckpt_name = f"ckpt/{ckpt_name}"
    ckpt = os.path.join(run_name, ckpt_name)

    inf_model = Sr3InferenceModel(
        config,
        ckpt,
        device,
        dtype,
        verbose=tqdm_verbose,
    )

    size = (
        (-1, *config.data.img_size)
        if (n_slices_per_sample is not None)
        else (
            # crop to max size of the model
            (max(config.data.img_size) - config.data.slice_thickness + 1,) * 3
            # if no crop_to_brain_margin is specified
            if config.data.crop_to_brain_margin is None
            else (tuple([config.data.img_size[0]] * 3) if crop else None)
        )
    )
    skull_strip = config.data.get("skull_strip", False)
    if skull_strip != 1 and skull_strip != 0:
        warnings.warn(
            f"Skull stripping should be either be on (1) or off (0) for the whole dataset, but is {skull_strip}. Rounding."
        )
        skull_strip = round(skull_strip)
    tar = [config.data.target_sequence] if config.data.target_sequence else []
    _, val = get_datasets(
        dataset=config.data.dataset,
        data_dir=config.data.data_dir,
        relevant_sequences=config.data.guidance_sequences + tar,
        size=size,
        cache=None,
        subset_train=0,
        normalize_to=(-1, 1),
        skull_strip=skull_strip,
        slicing_direction="axial",  # always axial, adapt matrix inside
        crop_to_brain_margin=config.data.crop_to_brain_margin,
    )
    slicing_direction = config.data.get("slicing_direction", "axial")

    for subject_index in subject_indices:
        if subject_index >= len(val):
            break

        vol = val[subject_index]
        if check_if_exists and vol["subject_ID"] in os.listdir(output_dir):
            print(
                f"Subject {vol['subject_ID']} already exists in {output_dir}. Skipping."
            )
            continue
        print(f"Saving subject {vol['subject_ID']} to {output_dir}.")

        # seed from subject ID for reproducibility across scripts / runs
        hash = seed_from_subject_id(vol["subject_ID"])
        torch.manual_seed(hash)

        guidance = torch.cat(  # C D H W
            [vol[seq] for seq in config.data.guidance_sequences], dim=0
        ).astype(dtype)

        if slicing_direction == "coronal":
            guidance = guidance.permute(0, 2, 1, 3)
        elif slicing_direction == "sagittal":
            guidance = guidance.permute(0, 3, 1, 2)

        if n_slices_per_sample is not None:
            # draw randomly n_slices_per_sample slices and predict on them
            indices = torch.randperm(guidance.shape[1])[:n_slices_per_sample]
            guidance = guidance[:, indices]  # C n_slices H W
            guidance = guidance.transpose(0, 1)  # transpose to n_slices C H W
            torch.manual_seed(hash + seed_offset)
            pred = inf_model.predict_slice(guidance).cpu().numpy()

            save_output_png(
                config.data.guidance_sequences + [config.data.target_sequence, "mask"],
                indices,
                output_dir,
                pred,
                vol,
                quantile_norm=quantile_norm,
            )

        else:
            torch.manual_seed(hash + seed_offset)
            pred = inf_model.synchronous_volume_denoising(guidance)

            # permute back to original orientation
            if slicing_direction == "coronal":
                vol["pred"] = pred.permute(0, 2, 1, 3)
            elif slicing_direction == "sagittal":
                vol["pred"] = pred.permute(0, 2, 3, 1)
            else:
                vol["pred"] = pred

            save_output_volume(
                vol,
                output_dir,
                save_keys=list(config.data.guidance_sequences)
                + ["pred", config.data.target_sequence, "mask"],
                target_sequence=config.data.target_sequence,
                quantile_norm=quantile_norm,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for running inference.")

    # model configs
    parser.add_argument(
        "--run_name",
        "-r",
        type=str,
        help="Name of the run (parent directory). The model checkpoint, config file, and the output will be taken from / saved to this directory.",
    )
    parser.add_argument(
        "--out_name",
        "-o",
        default="inference",
        type=str,
        help="Name to the output directory.",
    )
    parser.add_argument(
        "--dtype",
        "-d",
        type=str,
        default=None,
        help="Data type for inference (if none provided the train config will be used).",
    )
    parser.add_argument("--batch_size", "-b", type=int, default=32, help="Batch size.")
    parser.add_argument(
        "configs",
        nargs="*",
        type=str,
        help="Paths to configuration files, overwrites the model config from <run_name>/config.yml.",
    )
    parser.add_argument(
        "--steps", "-t", type=int, default=None, help="Number of steps."
    )
    parser.add_argument(  # see https://github.com/Project-MONAI/GenerativeModels/issues/397#issuecomment-1954514581
        "--timestep_spacing",
        "-ts",
        type=str,
        default=None,
        help="Timestep spacing (default: leading, option: linspace ie. forcing start at t=T) ",
    )

    # define checkpoint
    parser.add_argument(
        "--ckpt",
        "-c",
        type=str,
        default="last.pth",
        help="Path to the checkpoint file. If not provided, the last checkpoint from <run_name>/ckpt will be used.",
    )

    # slicing direction
    parser.add_argument(
        "--slicing_direction",
        "-sd",
        type=str,
        default=None,
        help="Slicing direction for the input data (default: from model config, usually axial).",
    )

    parser.add_argument(
        "--guidance_sequences",
        "-guid",
        nargs="+",
        type=str,
    )
    parser.add_argument(
        "--target_sequence",
        "-tar",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dataset_json",
        "-dj",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--data_dir",
        "-dd",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--omit_mask",
        "-om",
        action="store_true",
        help="Ignore/don't provide mask from defining the synthesis ROI.",
    )

    # Optional positional argument for start and end indices
    parser.add_argument(
        "--start", "-s", type=int, default=0, help="Start index for the validation set."
    )
    parser.add_argument(
        "--end", "-e", type=int, default=10000, help="End index for the validation set."
    )

    # tqdm verbosity
    parser.add_argument(
        "--no_verbose",
        "-nv",
        action="store_true",
    )

    # check if output already exists
    parser.add_argument(
        "--check_if_exists",
        "-ce",
        action="store_true",
    )

    # Optionally subset to n randomly drawn slices
    parser.add_argument(
        "--n_slices_per_sample",
        "-n",
        type=int,
        default=None,
        help="Number of slices per subject to subset.",
    )

    # optional_seed
    parser.add_argument(
        "--seed",
        "-seed",
        type=int,
        default=0,
        help="Change seed for the diffusion process to assess sampling stability/diversity.",
    )

    # quantile norm
    parser.add_argument(
        "--quantile_norm",
        "-q",
        type=bool,
        default=False,
        help="Whether to normalize the output by quantiles.",
    )

    parser.add_argument(
        "-f", "--force", action="store_true", help="Use the force, Luke."
    )

    args = parser.parse_args()

    run_name = args.run_name
    config = merge_cli_config(args, default_config=f"output/{run_name}/config.yml")

    # overwrite config with manual args
    if args.dtype is not None:
        config.trainer.dtype = args.dtype
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.steps is not None:
        config.model.num_inference_steps = args.steps
    if args.timestep_spacing is not None:
        config.model.train_noise_sched.timestep_spacing = args.timestep_spacing
    if args.slicing_direction is not None:
        config.data.slicing_direction = args.slicing_direction
    if args.guidance_sequences is not None:
        config.data.guidance_sequences = args.guidance_sequences
    if args.target_sequence is not None:
        config.data.target_sequence = args.target_sequence
    if args.dataset_json is not None:
        config.data.dataset = args.dataset_json
    if args.data_dir is not None:
        config.data.data_dir = args.data_dir
    if args.omit_mask:
        config.data.crop_to_brain_margin = None

    if args.force:
        print("YODA predicting your images is")
        print("Patience you must have, my young Padawan!")
        print("\n" + mini_yoda + "\n")

    print(
        f"predicting {config.data.target_sequence} from {config.data.guidance_sequences}"
    )

    # create output directory
    output_dir = os.path.join("output", run_name, args.out_name)
    if os.path.exists(output_dir):
        print(f"Output directory {output_dir} already exists.")
    os.makedirs(output_dir, exist_ok=True)

    subject_indices = list(range(args.start, args.end))

    main(
        args.run_name,
        output_dir,
        args.ckpt,
        config,
        subject_indices,
        args.n_slices_per_sample,
        args.seed,
        args.quantile_norm,
        not args.no_verbose,
        args.check_if_exists,
    )

    if not args.no_verbose:
        print(f"finished inference for {args.run_name}")
        print(f"output saved to {output_dir}")
        print(f"thanks for using YODA!")
    if args.force:
        print("\n" + the_force + "\n")

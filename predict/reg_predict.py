import warnings

import sys, os

sys.path.append(os.path.realpath(os.path.dirname(os.getcwd())))

import argparse
import torch
import numpy as np

from lib.datasets import get_datasets
from lib.inference import save_output_volume, RegressionInferer
from lib.utils.run_utils import merge_cli_config
from lib.utils.monai_helper import parse_dtype

device = "cuda"


def main(
    run_name,
    output_dir,
    ckpt_name,
    config,
    subject_indices,
    check_if_exists=False,
    view_agg=False,
    crop=False,
):
    dtype = parse_dtype(config.trainer.dtype)
    if not "/" in ckpt_name:
        ckpt_name = f"ckpt/{ckpt_name}"
    ckpt = os.path.join(run_name, ckpt_name)

    inf_model = RegressionInferer(
        config, ckpt=ckpt, device=device, dtype=parse_dtype(config.trainer.dtype)
    )
    slicing_directions = [config.data.get("slicing_direction", "axial")]
    if view_agg:
        if not config.data.get("random_slicing_direction", 0):
            warnings.warn(
                "Random slicing direction was not trained, might not work as expected."
            )
        slicing_directions = ["axial", "coronal", "sagittal"]

    size = tuple([config.data.img_size[0]] * 3) if crop else None
    skull_strip = config.data.get("skull_strip", False)
    if skull_strip != 1 and skull_strip != 0:
        warnings.warn(
            f"Skull stripping should be either be on (1) or off (0) for the whole dataset, but is {skull_strip}. Rounding."
        )
        skull_strip = round(skull_strip)
    tar = []
    if config.data.target_sequence:
        if isinstance(config.data.target_sequence, str):
            tar += [config.data.target_sequence]
        else:
            tar = list(config.data.target_sequence)
            if p := config.model.get("predicted_sequences", ""):
                # in case predicted sequences are defined in model config, add those as well
                tar = [s for s in p if s in tar]
                print(
                    f"Predicted sequences: {config.model.predicted_sequences}, target sequences: {config.data.target_sequence})"
                )

    _, val = get_datasets(
        dataset=config.data.dataset,
        data_dir=config.data.data_dir,
        relevant_sequences=list(config.data.guidance_sequences) + tar,
        size=size,
        crop_to_brain_margin=config.data.crop_to_brain_margin,
        cache=None,  # if config.data.cache != "persistent" else "persistent",
        subset_train=0,
        normalize_to=(-1, 1),
        skull_strip=skull_strip,
        slicing_direction="axial",  # always axial, adapt matrix inside
    )

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

        if config.trainer.get("alternate_target", False) == True:
            # always predict tar from src
            seqs = config.data.target_sequence
            if isinstance(seqs, str):
                seqs = [seqs]
            seqs += config.data.guidance_sequences
            seqs = sorted(list(set(seqs)))
            guidance_ax = torch.cat([vol[seq] for seq in seqs], dim=0).to(dtype)
            # set target to 0 , HOTFIX
            guidance_ax[seqs.index(config.data.target_sequence)] *= 0
        else:
            guidance_ax = torch.cat(  # C D H W
                [vol[seq] for seq in config.data.guidance_sequences], dim=0
            ).to(dtype)

        res = []
        for slicing_direction in slicing_directions:
            if slicing_direction == "coronal":
                guidance = guidance_ax.permute(0, 2, 1, 3)
            elif slicing_direction == "sagittal":
                guidance = guidance_ax.permute(0, 3, 1, 2)
            else:
                guidance = guidance_ax

            pred = inf_model(guidance)

            # permute back to original orientation
            if slicing_direction == "coronal":
                pred = pred.permute(0, 2, 1, 3)
            elif slicing_direction == "sagittal":
                pred = pred.permute(0, 2, 3, 1)
            else:
                pred = pred.permute(0, 1, 2, 3)
            res.append(pred.cpu())
        res = torch.stack(res, dim=0).mean(dim=0).float()  # average --> C D H W
        if res.shape[0] == 1:
            vol["pred"] = res
            save_pred = ["pred"]
        else:
            save_pred = []
            for i in range(res.shape[0]):
                if p := config.model.get("predicted_sequences", ""):
                    if len(p) == len(config.data.target_sequence):
                        vol[f"pred_{p[i]}"] = res[[i]]
                        save_pred.append(f"pred_{config.data.target_sequence[i]}")
                    else:
                        vol[f"pred_{p[i]}"] = res[[i]]
                        save_pred.append(f"pred_{p[i]}")
                else:
                    vol[f"pred_{i}"] = res[[i]]
                    save_pred.append(f"pred_{i}")

        tar_save_keys = []
        tar = config.data.target_sequence
        if isinstance(tar, str):  # defined as iterable (multiple targets)
            tar_save_keys = [tar]
        else:
            try:
                tar_save_keys = list(config.data.target_sequence)
            except:  # not defined at all
                if subject_index == subject_indices[0]:
                    print("Could not parse target sequence for saving.")

        save_output_volume(
            vol,
            output_dir,
            save_keys=list(config.data.guidance_sequences)
            + ["mask"]
            + save_pred
            + tar_save_keys,
            target_sequence=tar,
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
    # run and average all views
    parser.add_argument(
        "--view_agg",
        "-va",
        action="store_true",
        help="Aggregate the views (axial, coronal, sagittal) for the prediction.",
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
        nargs="+",
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

    # Optional positional argument for start and end indices
    parser.add_argument(
        "--start", "-s", type=int, default=0, help="Start index for the validation set."
    )
    parser.add_argument(
        "--end", "-e", type=int, default=10000, help="End index for the validation set."
    )

    # check if output already exists
    parser.add_argument(
        "--check_if_exists",
        "-ce",
        action="store_true",
    )

    args = parser.parse_args()

    if "/" not in args.run_name:
        args.run_name = "reg/" + args.run_name

    print(f"Running inference for {args.run_name}")

    run_name = args.run_name
    config = merge_cli_config(args, default_config=f"output/{run_name}/config.yml")

    # overwrite config with manual args
    if args.dtype is not None:
        config.trainer.dtype = args.dtype
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.slicing_direction is not None:
        config.data.slicing_direction = args.slicing_direction
    if args.guidance_sequences is not None:
        config.data.guidance_sequences = args.guidance_sequences
    if args.target_sequence is not None:
        if isinstance(args.target_sequence, list) and len(args.target_sequence) == 1:
            args.target_sequence = args.target_sequence[0]
        config.data.target_sequence = args.target_sequence
    if args.dataset_json is not None:
        config.data.dataset = args.dataset_json
    if args.data_dir is not None:
        config.data.data_dir = args.data_dir

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
        args.check_if_exists,
        args.view_agg,
    )

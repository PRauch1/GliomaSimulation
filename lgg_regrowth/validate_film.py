import argparse
import os

from eval.eval_io import build_case_context_film
from eval.eval_models import build_film_model
from eval.predict_common import apply_model_film
from eval.validate_common import full_validation_common

import constants

def parse_args():
    ap = argparse.ArgumentParser(description="FiLM full validation.")
    ap.add_argument("--data-root", type=str, default=constants.DIR_VALIDATION)
    ap.add_argument("--model-path", type=str, default=os.path.join(constants.DIR_MODEL, constants.MODEL_FILE_NAME))
    ap.add_argument("--threshold", type=float, default=constants.DEFAULT_THRESHOLD)
    ap.add_argument("--max-vox-samples-per-fu", type=int, default=constants.DEFAULT_MAX_VOX_SAMPLES)
    ap.add_argument("--global-out-dir", type=str, default=None)
    ap.add_argument("--no-save-predictions", action="store_true")
    ap.add_argument("--no-save-curves", action="store_true")
    ap.add_argument("--no-print-distinct-metadata", action="store_true")
    return ap.parse_args()


def predict_one_film(model, device, case_ctx, target_days: float, t_max_days: float, enforce_residual: bool = True):
    return apply_model_film(
        model=model,
        device=device,
        baseline=case_ctx["baseline"],
        residual=case_ctx["residual"],
        meta_cat=case_ctx["meta_cat"],
        target_days=target_days,
        t_max_days=t_max_days,
        enforce_residual=enforce_residual,
    )


def main():
    args = parse_args()

    model_bundle = build_film_model(
        device=None,
        model_path=args.model_path,
        data_root=args.data_root,
        print_distinct_metadata=not args.no_print_distinct_metadata,
    )

    full_validation_common(
        data_root=args.data_root,
        model_bundle=model_bundle,
        threshold=float(args.threshold),
        max_vox_samples_per_fu=int(args.max_vox_samples_per_fu),
        global_out_dir=args.global_out_dir,
        save_predictions=not args.no_save_predictions,
        save_curves=not args.no_save_curves,
        out_dir_name=constants.DIR_EVAL_OUTPUTS,
        nsd_tols_mm=constants.NSD_TOLS_MM,
        calibration_nbins=constants.CALIBRATION_NBINS,
        min_mask_voxels=constants.MIN_MASK_VOXELS,
        build_case_context_fn=build_case_context_film,
        predict_one_fn=predict_one_film,
        curve_stats_filename="curve_stats.json",
    )


if __name__ == "__main__":
    main()
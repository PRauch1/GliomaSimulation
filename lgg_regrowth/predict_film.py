import argparse
import os

from eval.predict_common import predict_cases_common
from eval.eval_models import build_film_model
from eval.eval_io import build_case_context_film
from eval.validate_common import maybe_compute_dice_at_target_day

import constants

def parse_args():
    ap = argparse.ArgumentParser(
        description="Prediction-only FiLM script (forecast mode only). Requires film.csv."
    )
    ap.add_argument("--target-days", type=float, required=True)
    ap.add_argument("--data-root", type=str, default=constants.DIR_VALIDATION)
    ap.add_argument("--model-path", type=str, default=os.path.join(constants.DIR_MODEL, constants.MODEL_FILE_NAME))
    ap.add_argument("--threshold", type=float, default=constants.DEFAULT_THRESHOLD)
    ap.add_argument("--no-enforce-residual", action="store_true")
    ap.add_argument("--global-out-dir", type=str, default=None)
    ap.add_argument("--no-save-predictions", action="store_true")
    ap.add_argument("--no-print-distinct-metadata", action="store_true")
    return ap.parse_args()


def predict_one_film(model, device, case_ctx, target_days: float, t_max_days: float, enforce_residual: bool = True):
    baseline = case_ctx["baseline"]
    residual = case_ctx["residual"]
    meta_cat = case_ctx["meta_cat"]
    return case_ctx["predict_fn"](
        model,
        device,
        baseline,
        residual,
        meta_cat,
        target_days,
        t_max_days,
        enforce_residual,
    )


def main():
    args = parse_args()

    threshold = float(args.threshold)
    enforce_residual = not args.no_enforce_residual
    save_predictions = not args.no_save_predictions
    print_distinct_metadata = not args.no_print_distinct_metadata

    model_bundle = build_film_model(
        device=None,
        model_path=args.model_path,
        data_root=args.data_root,
        print_distinct_metadata=print_distinct_metadata,
    )

    predict_cases_common(
        data_root=args.data_root,
        model_bundle=model_bundle,
        target_days=float(args.target_days),
        threshold=threshold,
        enforce_residual=enforce_residual,
        global_out_dir=args.global_out_dir,
        save_predictions=save_predictions,
        build_case_context_fn=build_case_context_film,
        predict_one_fn=predict_one_film,
        maybe_compute_dice_fn=maybe_compute_dice_at_target_day,
    )


if __name__ == "__main__":
    main()
# ================================================================
#  Evaluate the demo images with the app's own inference code
#
#  HOW TO RUN (from the repo root, with the project venv):
#    python scripts/eval_demo_images.py
#
#  The preprocessing, model-loading, and inference functions are
#  pulled straight out of dermascope/skin_cancer_app.py at runtime,
#  so this script always uses exactly the code the app uses. They
#  are extracted from the source instead of imported because
#  importing the app would run the whole Streamlit page.
# ================================================================

import ast
import csv
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_PATH   = os.path.join(REPO_ROOT, "dermascope", "skin_cancer_app.py")
DEMO_DIR   = os.path.join(REPO_ROOT, "demo_images")
LABELS_CSV = os.path.join(DEMO_DIR, "labels.csv")

APP_FUNCTIONS = ("load_model_and_labels", "preprocess_image", "run_inference")
APP_CONSTANTS = ("IMG_SIZE",)


def load_app_functions() -> dict:
    """Extract the app's functions and constants without running the Streamlit page."""
    with open(APP_PATH) as f:
        tree = ast.parse(f.read(), filename=APP_PATH)

    wanted = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in APP_FUNCTIONS:
            node.decorator_list = []  # drop @st.cache_resource; the body is untouched
            wanted.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in APP_CONSTANTS for t in node.targets
        ):
            wanted.append(node)

    namespace = {"__file__": APP_PATH, "np": np, "cv2": cv2, "Image": Image,
                 "os": os, "json": json}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), APP_PATH, "exec"), namespace)

    missing = [n for n in APP_FUNCTIONS + APP_CONSTANTS if n not in namespace]
    if missing:
        sys.exit(f"Could not find {missing} in {APP_PATH}")
    return namespace


def main():
    app = load_app_functions()

    interpreter, labels = app["load_model_and_labels"]()
    if interpreter is None:
        sys.exit("Model file not found next to the app.")

    with open(LABELS_CSV) as f:
        rows = list(csv.DictReader(f))

    print(f"{'filename':<18} {'true':<6} {'predicted':<10} {'confidence':>10}")
    print("-" * 48)

    correct = 0
    for row in rows:
        # Same steps as the app: open → RGB → preprocess → infer → argmax
        img_pil    = Image.open(os.path.join(DEMO_DIR, row["filename"])).convert("RGB")
        input_data = app["preprocess_image"](img_pil)
        probs      = app["run_inference"](interpreter, input_data)

        top_idx    = int(np.argmax(probs))
        predicted  = labels[str(top_idx)]
        confidence = float(probs[top_idx])

        is_correct = predicted == row["dx"]
        correct   += is_correct
        mark = "" if is_correct else "  ✗"
        print(f"{row['filename']:<18} {row['dx']:<6} {predicted:<10} {confidence:>10.1%}{mark}")

    print("-" * 48)
    print(f"Correct: {correct} / {len(rows)}")


if __name__ == "__main__":
    main()

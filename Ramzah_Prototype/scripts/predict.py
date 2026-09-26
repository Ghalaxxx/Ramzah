"""CLI: one isolated sign video -> actual checkpoint prediction and confidence."""

import argparse

from ramzah.cv import SignPredictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "video", help="MP4/WebM/MOV/AVI path containing one isolated sign"
    )
    args = parser.parse_args()
    result = SignPredictor().predict(args.video)
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Needs review: {str(result['needs_review']).lower()}")
    print("Alternatives:")
    for alternative in result["top_k"]:
        print(f"  - {alternative['word']}: {alternative['confidence']:.4f}")


if __name__ == "__main__":
    main()

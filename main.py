from src.perspective import adjust_perspective
import argparse


def parse_ratio(value: str):
    if value in ("a4", "frame"):
        return value
    try:
        ratio = float(value.replace(",", "."))
    except ValueError:
        raise argparse.ArgumentTypeError("use 'a4', 'frame' or a positive number (height/width).")
    if ratio <= 0:
        raise argparse.ArgumentTypeError("the ratio must be positive.")
    return ratio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adjust perspective of documents in a video based on ground truth data.")

    parser.add_argument("video_filepath", type=str, help="Path to the video file.")
    parser.add_argument("ground_truth_filepath", type=str, help="Path to the ground truth XML file.")
    parser.add_argument("output_folder", type=str, help="Path to the folder where the output images will be saved.")
    parser.add_argument("--n_sample", type=int, default=5, help="Number of random samples to process. Default is 5.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for the random sampling (reproducible runs).")
    parser.add_argument("--ratio", type=parse_ratio, default="a4",
                        help="Proportion of the output image: 'a4' (default, height/width = sqrt(2)), "
                             "'frame' (same size as the video frame, the previous behaviour) or a number (height/width).")
    parser.add_argument("--height", type=int, default=1080, help="Height of the output image in pixels. Default is 1080.")
    args = parser.parse_args()

    saved = adjust_perspective(args.video_filepath, args.ground_truth_filepath, args.output_folder, args.n_sample,
                               seed=args.seed, ratio=args.ratio, height=args.height)
    print(f"Saved {len(saved)} frame(s): {', '.join(saved)}")

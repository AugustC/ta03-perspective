from src.perspective import adjust_perspective
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adjust perspective of documents in a video based on ground truth data.")

    parser.add_argument("video_filepath", type=str, help="Path to the video file.")
    parser.add_argument("ground_truth_filepath", type=str, help="Path to the ground truth XML file.")
    parser.add_argument("output_folder", type=str, help="Path to the folder where the output images will be saved.")
    parser.add_argument("--n_sample", type=int, default=5, help="Number of random samples to process. Default is 5.")
    args = parser.parse_args()

    adjust_perspective(args.video_filepath, args.ground_truth_filepath, args.output_folder, args.n_sample)

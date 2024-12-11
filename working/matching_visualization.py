import h5py
import cv2
import numpy as np
from pathlib import Path
import random

def visualize_matches_from_h5(
    matches_path, keypoints_path, image_dir, output_dir, num_matches=30, max_image_pairs=10
):
    from pathlib import Path
    import cv2
    import numpy as np
    import h5py
    import random

    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all images in the directory
    image_files = {img.stem: img for img in Path(image_dir).glob("*.png")}  # Use .stem for file name without suffix

    # Open matches.h5 and keypoints.h5 files
    pair_count = 0  # Counter to limit the number of processed image pairs
    with h5py.File(matches_path, "r") as matches_f, h5py.File(keypoints_path, "r") as keypoints_f:
        for img1_name_with_ext in matches_f.keys():
            img1_name = Path(img1_name_with_ext).stem  # Remove extension
            img1_path = image_files.get(img1_name)
            if img1_path is None or not img1_path.exists():
                print(f"Image not found: {img1_name}")
                continue

            img1 = cv2.imread(str(img1_path))

            # Retrieve keypoints for img1
            if img1_name_with_ext not in keypoints_f:
                print(f"Keypoints not found for image: {img1_name_with_ext}")
                continue
            kp1 = np.array(keypoints_f[img1_name_with_ext])

            for img2_name_with_ext in matches_f[img1_name_with_ext].keys():
                img2_name = Path(img2_name_with_ext).stem  # Remove extension
                img2_path = image_files.get(img2_name)
                if img2_path is None or not img2_path.exists():
                    print(f"Image not found: {img2_name}")
                    continue

                img2 = cv2.imread(str(img2_path))

                # Retrieve keypoints for img2
                if img2_name_with_ext not in keypoints_f:
                    print(f"Keypoints not found for image: {img2_name_with_ext}")
                    continue
                kp2 = np.array(keypoints_f[img2_name_with_ext])

                # Load matches
                matches = np.array(matches_f[img1_name_with_ext][img2_name_with_ext])

                # Randomly sample matches if necessary
                if len(matches) > num_matches:
                    matches = matches[np.random.choice(len(matches), num_matches, replace=False)]

                # Concatenate images
                img1_h, img1_w = img1.shape[:2]
                img2_h, img2_w = img2.shape[:2]
                max_h = max(img1_h, img2_h)
                concat_img = np.zeros((max_h, img1_w + img2_w, 3), dtype=img1.dtype)
                concat_img[:img1_h, :img1_w, :] = img1
                concat_img[:img2_h, img1_w:, :] = img2

                # Draw matches
                for pt1_idx, pt2_idx in matches:
                    # Get the coordinates of the keypoints
                    try:
                        x1, y1 = int(kp1[pt1_idx][0]), int(kp1[pt1_idx][1])
                        x2, y2 = int(kp2[pt2_idx][0]) + img1_w, int(kp2[pt2_idx][1])  # Offset x2 by img1 width
                    except IndexError:
                        print(f"IndexError: Skipping match ({pt1_idx}, {pt2_idx})")
                        continue
                    color = tuple(np.random.randint(0, 255, size=3).tolist())  # Random color
                    cv2.line(concat_img, (x1, y1), (x2, y2), color, 2)  # Draw line
                    cv2.circle(concat_img, (x1, y1), 5, color, -1)  # Draw keypoint on img1
                    cv2.circle(concat_img, (x2, y2), 5, color, -1)  # Draw keypoint on img2

                # Save visualization
                output_file = output_dir / f"matches_{img1_name}_{img2_name}.png"
                cv2.imwrite(str(output_file), concat_img)  # Save the image directly using OpenCV
                print(f"Saved: {output_file}")

                pair_count += 1
                if pair_count >= max_image_pairs:
                    print(f"Reached the limit of {max_image_pairs} image pairs. Stopping.")
                    return



# Usage Example
matches_path = "../output/.feature_outputs/church_church_trial0/colmap_rec_aliked_refine/0"  # Path to matches.h5
image_dir = "../input/image-matching-challenge-2024/train/church/images"  # Directory containing images
output_dir = "../output/matching"  # Directory to save visualizations
keypoints_path = "../output/.feature_outputs/church_church_trial0/keypoints.h5"

visualize_matches_from_h5(matches_path, keypoints_path, image_dir, output_dir)


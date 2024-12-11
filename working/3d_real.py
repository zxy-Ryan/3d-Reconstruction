import rerun as rr
from numpy.random import default_rng

rr.init("IMC_church", spawn=True) # rerun's init method

import io
import os
import re
import zipfile
from argparse import ArgumentParser
from pathlib import Path
from typing import Final

import cv2
import numpy as np
import numpy.typing as npt
import requests
import pycolmap

from tqdm import tqdm

sfm_path = '../output/.feature_outputs/church_church_trial0/colmap_rec_aliked_refine/0'
images_path = '../input/image-matching-challenge-2024/train/church/images'

DESCRIPTION = "3D Sparse Reconstruction"

from scipy.spatial.transform import Rotation as R
from rerun.datatypes import Angle, RotationAxisAngle

FILTER_MIN_VISIBLE = 50
def scale_camera(camera, resize: tuple[int, int]) -> tuple[pycolmap.Camera, npt.NDArray[np.float_]]:
    """Scale the camera intrinsics to match the resized image."""
    assert camera.model == "PINHOLE"
    new_width = resize[0]
    new_height = resize[1]
    scale_factor = np.array([new_width / camera.width, new_height / camera.height])

    # For PINHOLE camera model, params are: [focal_length_x, focal_length_y, principal_point_x, principal_point_y]
    new_params = np.append(camera.params[:2] * scale_factor, camera.params[2:] * scale_factor)

    return (pycolmap.Camera(camera.id, camera.model, new_width, new_height, new_params), scale_factor)



def read_and_log_sparse_reconstruction(rec_path: Path, img_path: Path, filter_output: bool, resize: tuple[int, int] | None) -> None:
    print("Reading sparse COLMAP reconstruction")
    reconstruction = pycolmap.Reconstruction(rec_path)
    cameras = reconstruction.cameras
    images = reconstruction.images
    points3D = reconstruction.points3D
    print("Building visualization by logging to Rerun")

    if filter_output:
        # Filter out noisy points
        points3D = {id: point for id, point in points3D.items() if point.color.any() and len(point.image_ids) > 2}

    rr.log("description", rr.TextDocument(DESCRIPTION, media_type=rr.MediaType.MARKDOWN), timeless=True)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, timeless=True)
    rr.log("plot/avg_reproj_err", rr.SeriesLine(color=[240, 45, 58]), timeless=True)

    # Iterate through images (video frames) logging data related to each frame.
    ii=0
    for image in tqdm(sorted(images.values(), key=lambda im: im.name)):  # type: ignore[no-any-return]
        image_file = img_path / image.name.replace('.jpg', '.png')
        if not os.path.exists(image_file):
            continue
        #print (image_file)

        # COLMAP sets image ids that don't match the original video frame
        idx_match = re.search(r"\d+", image.name)
        assert idx_match is not None
        frame_idx = int(idx_match.group(0))


        camera = cameras[image.camera_id]
        if resize:
            camera, scale_factor = scale_camera(camera, resize)
        else:
            scale_factor = np.array([1.0, 1.0])

        visible_ids = [id_ for id_ in points3D.keys() if image.has_point3D(id_) ]


        if filter_output and len(visible_ids) < FILTER_MIN_VISIBLE:
            continue

        visible_xyzs = [points3D[idx] for idx in visible_ids]
        visible_xys = np.array([x.xy for x in image.get_observation_points2D()])
        if resize:
            visible_xys *= scale_factor

        rr.set_time_sequence("frame", frame_idx)
        try:
            points = [point.xyz for point in visible_xyzs]
        except Exception as e:
            print (e)
            continue
        point_colors = [point.color for point in visible_xyzs]
        point_errors = [point.error for point in visible_xyzs]

        rr.log("plot/avg_reproj_err", rr.Scalar(np.mean(point_errors)))

        rr.log("points", rr.Points3D(points, colors=point_colors), rr.AnyValues(error=point_errors))

        # COLMAP's camera transform is "camera from world"
        rr.log(
            "camera", rr.Transform3D(translation=image.cam_from_world.translation,
                                     rotation=rr.Quaternion(xyzw=image.cam_from_world.rotation.quat), from_parent=True)
        )
        rr.log("camera", rr.ViewCoordinates.RDF, timeless=True)  # X=Right, Y=Down, Z=Forward

        # Log camera intrinsics
        assert str(camera.model) in  ["CameraModelId.SIMPLE_PINHOLE", "CameraModelId.PINHOLE"]
        if str(camera.model) == "CameraModelId.SIMPLE_PINHOLE":
            rr.log(
                "camera/image",
                rr.Pinhole(
                    resolution=[camera.width, camera.height],
                    focal_length=[camera.params[0], camera.params[0]],
                    principal_point=camera.params[1:],
                ),
            )
        else:
            rr.log(
                "camera/image",
                rr.Pinhole(
                    resolution=[camera.width, camera.height],
                    focal_length=camera.params[:2],
                    principal_point=camera.params[2:],
                ),
            )
        if resize:
            bgr = cv2.imread(str(image_file))
            bgr = cv2.resize(bgr, resize)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rr.log("camera/image", rr.Image(rgb).compress(jpeg_quality=75))
        else:
            rr.log("camera/image", rr.ImageEncoded(path=img_path / image.name.replace('.jpg', '.png')))
        rr.log("camera/image/keypoints", rr.Points2D(visible_xys, colors=[34, 138, 167]))
    print ("Now preparing visualization engine")


def log_combined_3d_points(rec_path: Path, resize=None, axis_length=0.5, image_plane_visual_scale=0.5):
    print("Reading sparse COLMAP reconstruction")
    reconstruction = pycolmap.Reconstruction(rec_path)
    cameras = reconstruction.cameras
    images = reconstruction.images
    points3D = reconstruction.points3D
    print("Building visualization by logging to Rerun")

    # Combine all 3D points
    all_points = []
    all_colors = []
    for point in points3D.values():
        all_points.append(point.xyz)
        all_colors.append(point.color)

    # Log all 3D points as a single entity
    rr.log("final_points", rr.Points3D(all_points, colors=all_colors))
    print(f"Logged {len(all_points)} 3D points.")

    # Log all cameras and their image planes
    for image in images.values():
        camera = cameras[image.camera_id]

        # Camera position and orientation
        position = image.cam_from_world.translation

        rr.log(
            f"camera_{image.image_id}", rr.Transform3D(
                translation=position,
                rotation=rr.Quaternion(xyzw=image.cam_from_world.rotation.quat),
                from_parent=True,
                axis_length=0
            )
        )

        # Log camera intrinsics (image plane) using built-in methods
        if str(camera.model) == "CameraModelId.SIMPLE_PINHOLE":
            rr.log(
                f"camera_{image.image_id}/image",
                rr.Pinhole(
                    resolution=[camera.width, camera.height],
                    focal_length=[camera.params[0], camera.params[0]],
                    principal_point=camera.params[1:],
                    image_plane_distance=0.3
                ),
            )
        else:
            rr.log(
                f"camera_{image.image_id}/image",
                rr.Pinhole(
                    resolution=[camera.width, camera.height],
                    focal_length=[camera.params[0], camera.params[1]],
                    principal_point=camera.params[2:],
                ),
            )

    print(f"Logged {len(images)} cameras and image planes.")



# Whatever you want to visualize in notebook, you should start the rec = rr.memory_recording()
rec = rr.memory_recording()
read_and_log_sparse_reconstruction(Path(sfm_path), Path(images_path), filter_output=False, resize=None)
# log_combined_3d_points(Path(sfm_path))

rr.spawn()


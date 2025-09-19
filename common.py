import os
import numpy as np
import numpy.typing as npt
import pycolmap as col
from collections.abc import Iterable
from scipy.spatial import KDTree

class Image:
    """An abstraction over the ``Image`` class of ``pycolmap``."""
    def __init__(self, path: str, image: col.Image):
        self.path = path
        self.name = os.path.split(image.name)[1]
        self.width = image.camera.width
        self.height = image.camera.height
        focal_length_x = image.camera.params[0]
        self.__half_field_of_view_x = np.arctan(self.width / (2 * focal_length_x))
        self.__image = image
        self.__camera = image.camera
    
    def is_in_image(self, point: npt.NDArray[np.float64]) -> bool:
        """Check if ``point`` is inside the image."""
        return (point[0] >= 0 and point[0] <= self.width and
                point[1] >= 0 and point[1] <= self.height)

    def is_in_x_field_of_view(self, point: npt.NDArray[np.float64]) -> bool:
        """Check if ``point`` is inside the field of view of the camera along the x axis.
        
        ``point`` must be in the camera frame.
        """
        angle = np.arctan2(point[0], point[2])
        return abs(angle) <= self.__half_field_of_view_x

    def cam_from_world(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Transform ``points`` from world frame to camera frame."""
        return self.__image.cam_from_world() * points

    def project_points_from_cam(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Project ``points`` to the image plane.
        
        ``points`` must be in the camera frame.

        The resulting array will contain vectors of NaN in correspondence of points that are behind the camera.
        """
        return self.__camera.img_from_cam(points)


purple = np.array([178, 0, 254], dtype=np.float64)

def filter_images(rec: col.Reconstruction, image_folder_path: str, image_filter: str) -> Iterable[Image]:
    """Get from ``rec`` the images containing in their name ``image_filter``.
    
    ``image_folder_path`` is the path where all images are stored.

    If no images match the filter stop the script.
    """
    images = []
    
    for image in rec.images.values():
        image: col.Image
        if image_filter in image.name:
            images.append(Image(image_folder_path + image.name, image))
    
    if not images:
        print(f"No image match the filter: {image_filter}")
        exit()

    return images

def get_valid_points_indices(image: Image, points_in_cam_frame: npt.NDArray[np.float64], projected_points: npt.NDArray[np.float64], neighbor_radius: float, max_depth_difference: float) -> npt.NDArray[np.int32]:
    """Get the indices of the points that are visible on ``image``.
    
    ``points_in_cam_frame`` is the numpy array of the points coordinates in the camera frame.
    
    ``projected points`` is the numpy array of the projected points coordinates on the image plane.

    The indices of ``points_in_cam_frame`` and ``projected_points`` must match, that is: 
    the projection of the n-th point of ``points_in_cam_frame`` is the n-th point of ``projected_points``.

    ``neighbor_radius`` specify the size of the neighborhood for the visibiliy check, that is what is the maximum distance between two neighbors.

    ``max_depth_difference`` specify the maximum difference that there can be between a point's distance from the camera and
    the minimum distance from the camera in its neighborhood to consider the point visible in ``image``.
    """
    # get the indices of points that:
    #     are not behind the camera
    #     are not outside of the image
    #     do not form an angle with the viewing direction greater than half fov on the x axis
    valid_points_indices = np.array([index for index in range(len(points_in_cam_frame))
                                        if  not np.isnan(projected_points[index][0]) and
                                            image.is_in_image(projected_points[index]) and
                                            image.is_in_x_field_of_view(points_in_cam_frame[index])]
                                    , dtype=np.int32)
    
    # get the distance from the camera of the valid points
    valid_points_depths = points_in_cam_frame[valid_points_indices][:,2]
    
    not_visible_points_indices = []
    projected_points_tree = KDTree(projected_points[valid_points_indices])
    neighbors = projected_points_tree.query_ball_tree(projected_points_tree, neighbor_radius)
    for index, neighbor_indices in enumerate(neighbors):
        lowest_depth = np.min(valid_points_depths[neighbor_indices], initial=np.inf)

        # if in the neighbor there is a point closer to the camera by a certain difference consider the point not visible
        if (lowest_depth != np.inf) and (valid_points_depths[index] - lowest_depth > max_depth_difference):
            # "index" refer to the index of the point in the tree where we already removed some of the projected points
            # so to get the original index (the one that can be used in "projected_points") we have to refer to "valid_points_indices"
            not_visible_points_indices.append(valid_points_indices[index])
    
    return np.setdiff1d(valid_points_indices, not_visible_points_indices)
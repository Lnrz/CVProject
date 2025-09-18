import os
import numpy as np
import numpy.typing as npt
import pycolmap as col
from collections.abc import Iterable
from scipy.spatial import KDTree

class Image:
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
        return (point[0] >= 0 and point[0] <= self.width and
                point[1] >= 0 and point[1] <= self.height)

    # the point is supposed to be in the camera frame
    def is_in_x_field_of_view(self, point: npt.NDArray[np.float64]) -> bool:
        angle = np.arctan2(point[0], point[2])
        return abs(angle) <= self.__half_field_of_view_x

    def cam_from_world(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self.__image.cam_from_world() * points

    def project_points_from_cam(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self.__camera.img_from_cam(points)


purple = np.array([178, 0, 254], dtype=np.float64)

def filter_images(rec: col.Reconstruction, image_folder_path: str, image_filter: str) -> Iterable[Image]:
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

        # skip point if in the neighbor there is a point closer to the camera by a certain difference
        if (lowest_depth != np.inf) and (valid_points_depths[index] - lowest_depth > max_depth_difference):
            # "index" refer to the index of the point in the tree where we already removed some of the projected points
            # so to get the original index (the one that can be used in "projected_points") we have to refer to "valid_points_indices"
            not_visible_points_indices.append(valid_points_indices[index])
    
    return np.setdiff1d(valid_points_indices, not_visible_points_indices)
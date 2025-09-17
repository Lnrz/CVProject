import os
import numpy as np
import numpy.typing as npt
import pycolmap as col
from collections.abc import Iterable

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
import numpy as np
import numpy.typing as npt
import pycolmap as col
import argparse as argp
from scipy.spatial import KDTree
from enum import Enum
from collections.abc import Iterable

class Format(Enum):
    SPARSE = 0
    DENSE = 1

class Image:
    def __init__(self, path: str, image: col.Image):
        self.path = path
        self.width = image.camera.width
        self.height = image.camera.height
        self.__image = image
        self.__camera = image.camera
    
    def project_points(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self.__camera.img_from_cam(self.__image.cam_from_world() * points)



def parse_arguments() -> argp.Namespace:
    parser = argp.ArgumentParser(description="Project Image Color to Model")
    parser.add_argument("reconstruction_path", type=str, help="Path to the reconstruction.")
    parser.add_argument("image_folder_path", type=str, help="Path where the images are stored.")
    parser.add_argument("image_name_filter", type=str, help="Only images with this filter in their names will be used.")
    parser.add_argument("output_path", type=str, help="Path where to write the output.")
    parser.add_argument("-mp", "--model_path", type=str, help="Path to the model. Required for dense flag.")
    parser.add_argument("-nd", "--neighbor_distance", type=float, default=15,
                        help="When processing a point projected on the image, this specifies the maximum distance of its neighbors. "
                             "Measured in pixel.")
    parser.add_argument("-mdd", "--max_depth_difference", type=float, default=1,
                        help="When processing a point projected on the image, "
                             "this specifies the maximum distance that there can be between the point's distance from the camera "
                             "and the distance from the camera of its neighbor that is nearest to the camera. "
                             "If greater the point will be skipped. "
                             "Measured in the unit of measure of the reconstruction/model")
    parser.add_argument("-s", "--sparse", action="store_true", help="Flag to output a reconstruction. Default.")
    parser.add_argument("-d", "--dense", action="store_true", help="Flag to output a model.")
    args = parser.parse_args()
    
    if args.dense and args.model_path is None:
        print("For the dense output you need to specify the model.")
        exit()

    return args

def project_colors(points: npt.NDArray[np.float64], images: Iterable[Image], neighbor_distance: float, max_depth_difference: float) -> npt.NDArray[np.uint8]:
    # data structure for keeping the colors of every point
    colors_per_point = [[] for i in range(points.shape[0])]
    # projecting colors to points
    print("Processing images...")
    for image_index, image in enumerate(images):
        image: Image
        print("    Processing " + str(image_index + 1) + "th image...")
        print("    Image is", image.path)
        
        # load image in memory
        # the array has size [height, width, 3]
        image_data = col.Bitmap.read(image.path, True).to_array()

        # project points on image
        projected_points = image.project_points(points)

        # calculate boolean array checking for nans (i.e. points behind the camera)
        is_not_behind_camera = np.logical_not(np.isnan(projected_points[:,0]))

        # put valid points in a 2dtree
        points_tree = KDTree(projected_points[is_not_behind_camera])
        
        for projected_point_index, projected_point in enumerate(projected_points):
            # check if point is behind the camera, if it is skip it
            if not is_not_behind_camera[projected_point_index]:
                continue

            # get the point depth in 3D
            point_depth = points[projected_point_index][2]
            
            # if the point is projected outside the visible part of the image ignore it
            if (projected_point[0] < 0 or projected_point[1] < 0  or
                projected_point[0] > image.width or projected_point[1] > image.height):
                continue
            
            # get the indices of the neighbors
            neighbor_indices = points_tree.query_ball_point(projected_point, neighbor_distance)
            
            # get the neighbor depths
            neighbor_depths = points[neighbor_indices][:,2]
            
            # get the depth of the neighbor nearest to the camera
            lowest_depth = np.min(neighbor_depths, initial=np.inf)

            # if it can find a neighbor closer to the camera by a certain difference
            # assume that this point is not actually seen in the image
            if (lowest_depth != np.inf) and (point_depth - lowest_depth > max_depth_difference):
                continue
            
            # get color from image
            color = image_data[np.floor(projected_point[1]).astype(np.int32), np.floor(projected_point[0]).astype(np.int32)] # TODO do bilinear or trilinear interpolation

            # add color to point
            colors_per_point[projected_point_index].append(color)

        break #for debugging purpouse

    # TODO to color not colored point median filter on every channel of HSV or vector median filter
    
    # calculate the color of the points
    purple = np.array([178, 0, 254], dtype=np.uint8)
    return np.array([np.mean(colors, axis=0).astype(np.uint8) # TODO mean, trimmed mean, median
                            if len(colors) > 0
                            else purple # color not colored points purple for easy detection
                            for colors in colors_per_point])

def write_rec(out_path: str, rec: col.Reconstruction, ids: npt.NDArray[np.uint32], colors: npt.NDArray[np.uint8]):
    # color reconstruction points
    print("Coloring...")
    for point_id, point_color in zip(ids, colors):
        rec.point3D(point_id).color = point_color
    
    # write reconstruction
    print("Writing reconstruction...")
    rec.write(out_path)

def write_ply(out_path: str, points: npt.NDArray[np.float64], colors: npt.NDArray[np.uint8]):
    # add points to model
    print("Coloring...")
    ply = col.Reconstruction()
    dummy_track = col.Track()
    for index in range(points.shape[0]):
        ply.add_point3D(points[index], dummy_track, colors[index])

    # write ply file
    print("Writing ply...")
    ply.export_PLY(out_path)


def main():
    # get args
    args = parse_arguments()
    rec_path = args.reconstruction_path
    image_folder_path = args.image_folder_path
    image_filter = args.image_name_filter
    out_path = args.output_path
    model_path = args.model_path
    neighbor_distance = args.neighbor_distance
    max_depth_difference = args.max_depth_difference
    out_format = Format.DENSE if args.dense else Format.SPARSE

    print("Loading data...")
    # load reconstruction
    rec = col.Reconstruction(rec_path)

    match out_format:
        case Format.SPARSE:
            # extract points and ids from rec
            ids = np.array([point_id for point_id in rec.points3D.keys()], dtype=np.uint32)
            points = np.array([point.xyz for point in rec.points3D.values()], dtype=np.float64)
        case Format.DENSE:
            # load model
            ply = col.Reconstruction()
            ply.import_PLY(model_path)

            # extract points from model
            points = np.array([point.xyz for point in ply.points3D.values()], dtype=np.float64)
            
            # unload model
            del ply
    
    # get images of interest
    images = []
    for image in rec.images.values():
        image: col.Image
        if image_filter in image.name:
            images.append(Image(image_folder_path + image.name, image))
    
    colors = project_colors(points, images, neighbor_distance, max_depth_difference)

    match out_format:
        case Format.SPARSE:
            write_rec(out_path, rec, ids, colors)
        case Format.DENSE:
            write_ply(out_path, points, colors)

if __name__ == "__main__":
    main()
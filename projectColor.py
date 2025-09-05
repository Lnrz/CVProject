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



def get_arguments() -> argp.Namespace:
    parser = argp.ArgumentParser(description="Project Image Color to Model")
    parser.add_argument("reconstruction_path", type=str, help="Path to the reconstruction.")
    parser.add_argument("image_folder_path", type=str, help="Path where the images are stored.")
    parser.add_argument("image_name_filter", type=str, help="Only images with this filter in their names will be used.")
    parser.add_argument("output_path", type=str, help="Path where to write the output.")
    parser.add_argument("-mp", "--model_path", type=str, help="Path to the model. Required for dense flag.")
    parser.add_argument("-nd", "--neighbor_distance", type=float, default=15,
                        help="When processing a point projected on the image, this specifies the maximum distance of its neighbors. "
                             "Measured in pixel. Default to 15.")
    parser.add_argument("-mdd", "--max_depth_difference", type=float, default=1,
                        help="When processing a point projected on the image, "
                             "this specifies the maximum distance that there can be between the point's distance from the camera "
                             "and the distance from the camera of its neighbor that is nearest to the camera. "
                             "If greater the point will be skipped. Default to 1. "
                             "Measured in the unit of measure of the reconstruction/model")
    parser.add_argument("-s", "--sparse", action="store_true", help="Flag to output a reconstruction. Default.")
    parser.add_argument("-d", "--dense", action="store_true", help="Flag to output a model.")
    parser.add_argument("-df", "--disable_filling", action="store_true", help="Flag to disable the filling process for not colored points.")
    parser.add_argument("-fnd", "--filling_neighbor_distance", type=float, default=1,
                        help="When filling not colored points, this specifies the maximum distance of its neighbors. "
                             "Measured in the unit of measure of the reconstruction/model. Default to 1.")
    parser.add_argument("-ft", "--filling_threshold", type=float, default=0.5,
                        help="The maximum percentage of not colored neighbors a not colored point can have to be filled. "
                              "If the percentage is exceeded the point is not filled. Default to 0.5.")
    parser.add_argument("-ic", "--image_cap", type=int, default=0, help="Maximum number of images to process. Useful for debugging. 0 means no cap.")
    args = parser.parse_args()
    
    if args.sparse and args.dense:
        print("You can only choose one between sparse and dense.")
        exit()
    if args.dense and args.model_path is None:
        print("For the dense output you need to specify the model.")
        exit()
    
    if args.neighbor_distance < 0:
        print("Neighbor distance can't be lower than 0.")
        exit()
    if args.max_depth_difference < 0:
        print("Max depth difference can't be lower than 0.")
        exit()
    
    if not args.disable_filling and args.filling_neighbor_distance < 0:
        print("Filling neighbor distance can't be lower than 0.")
        exit()
    if not args.disable_filling and (args.filling_threshold < 0 or args.filling_threshold >= 1): 
        print("Filling threshold must be in [0,1)")
        exit()
    
    if args.image_cap < 0:
        print("Image cap must be at least 0.")
        exit()

    return args

def filter_images(rec: col.Reconstruction, image_folder_path: str, image_filter: str) -> Iterable[Image]:
    images = []
    
    for image in rec.images.values():
        image: col.Image
        if image_filter in image.name:
            images.append(Image(image_folder_path + image.name, image))
    
    if len(images) == 0:
        print(f"No image match the filter: {image_filter}")
        exit()

    return images

def project_colors(points: npt.NDArray[np.float64], images: Iterable[Image], neighbor_distance: float, max_depth_difference: float, disable_filling: bool, filling_threshold: float, filling_radius: float, image_cap: int) -> npt.NDArray[np.uint8]:
    # data structure for keeping the colors of every point
    colors_per_point = [[] for i in range(points.shape[0])]
    # projecting colors to points
    print("Processing images...")
    images_length = len(images)
    for image_index, image in enumerate(images):
        image: Image
        image_index_str = str(image_index + 1)
        print(f"    Processing {image_index_str}th image out of {images_length}...")
        print(f"    Image is {image.path}")
        
        # load image in memory
        # the array has size [height, width, 3]
        image_data = col.Bitmap.read(image.path, True)
        if image_data is None:
            print(f"    Couldn't read image at {image.path}. Skipping it...")
            continue
        image_data = image_data.to_array()

        # project points on image
        projected_points = image.project_points(points)

        # calculate boolean array checking for nans (i.e. points behind the camera)
        is_behind_camera = np.isnan(projected_points[:,0])

        # put valid points in a 2dtree
        projected_points_tree = KDTree(projected_points[np.logical_not(is_behind_camera)])
        
        print(f"    Projecting {image_index_str}th image's colors...")
        for projected_point_index, projected_point in enumerate(projected_points):
            # check if point is behind the camera, if it is skip it
            if is_behind_camera[projected_point_index]:
                continue

            # get the point depth in 3D
            point_depth = points[projected_point_index][2]
            
            # if the point is projected outside the visible part of the image ignore it
            if (projected_point[0] < 0 or projected_point[1] < 0  or
                projected_point[0] > image.width or projected_point[1] > image.height):
                continue
            
            # get the indices of the neighbors
            neighbor_indices = projected_points_tree.query_ball_point(projected_point, neighbor_distance)
            
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
        
        if image_cap != 0 and image_index == image_cap - 1:
            break

    print("Calculating colors...")
    # color for not colored points
    purple = np.array([178, 0, 254], dtype=np.float64)
    # calculate final color for colored points
    colors = np.array([np.mean(colors, axis=0)
                            if colors
                            else purple
                            for colors in colors_per_point])
    
    if not disable_filling:
        print("Filling...")
        # get indices of not colored points
        not_colored_point_indices = np.array([index
                                                for index, colors in enumerate(colors_per_point)
                                                if not colors])
        # put points in a 3D tree
        points_tree = KDTree(points)
        
        for index in not_colored_point_indices:
            # get neighbor indices
            neighbor_indices = points_tree.query_ball_point(points[index], filling_radius)
            # check that there are neighbors otherwise skip the point
            if not neighbor_indices:
                continue
            
            # get the indices of not colored neighbors
            not_colored_neighbor_indices = np.intersect1d(neighbor_indices, not_colored_point_indices)
            # if the percentage of not colored neighbors is greater than the threshold skip the point
            if len(not_colored_neighbor_indices) / len(neighbor_indices) > filling_threshold:
                continue
            
            # remove purple neighbor indices
            neighbor_indices = np.setdiff1d(neighbor_indices, not_colored_neighbor_indices)
            # get the neighbor colors
            neighbor_colors = colors[neighbor_indices]
            
            # put as the color of the point the median of the neighbor colors
            colors[index] = np.median(neighbor_colors, axis=0)
    
    return colors.astype(np.uint8)

def write_rec(out_path: str, rec: col.Reconstruction, ids: npt.NDArray[np.uint32], colors: npt.NDArray[np.uint8]):
    # color reconstruction points
    print("Coloring...")
    for point_id, point_color in zip(ids, colors):
        rec.point3D(point_id).color = point_color
    
    print("Writing reconstruction...")
    rec.write(out_path)

def write_ply(out_path: str, points: npt.NDArray[np.float64], colors: npt.NDArray[np.uint8]):
    # add points to model
    print("Coloring...")
    ply = col.Reconstruction()
    dummy_track = col.Track()
    for index in range(points.shape[0]):
        ply.add_point3D(points[index], dummy_track, colors[index])

    print("Writing ply...")
    ply.export_PLY(out_path)


def main():
    args = get_arguments()
    rec_path = args.reconstruction_path
    image_folder_path = args.image_folder_path
    image_filter = args.image_name_filter
    out_path = args.output_path
    model_path = args.model_path
    neighbor_distance = args.neighbor_distance
    max_depth_difference = args.max_depth_difference
    out_format = Format.DENSE if args.dense else Format.SPARSE
    disable_filling = args.disable_filling
    filling_radius = args.filling_neighbor_distance
    filling_threshold = args.filling_threshold
    image_cap = args.image_cap

    print("Loading data...")
    rec = col.Reconstruction(rec_path)

    match out_format:
        case Format.SPARSE:
            ids = np.array([point_id for point_id in rec.points3D.keys()], dtype=np.uint32)
            points = np.array([point.xyz for point in rec.points3D.values()], dtype=np.float64)
        case Format.DENSE:
            ply = col.Reconstruction()
            ply.import_PLY(model_path)

            points = np.array([point.xyz for point in ply.points3D.values()], dtype=np.float64)
            
            # unload model
            del ply

    if len(points) == 0:
        print(f"No points found in {"reconstruction" if out_format == Format.SPARSE else "model"}.")
        exit()

    images = filter_images(rec, image_folder_path, image_filter)
    
    colors = project_colors(points, images, neighbor_distance, max_depth_difference, disable_filling, filling_threshold, filling_radius, image_cap)

    match out_format:
        case Format.SPARSE:
            write_rec(out_path, rec, ids, colors)
        case Format.DENSE:
            write_ply(out_path, points, colors)

if __name__ == "__main__":
    main()
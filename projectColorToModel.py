import numpy as np
import numpy.typing as npt
import pycolmap as col
import argparse as argp
from scipy.spatial import KDTree
from enum import Enum
from common import Image, purple, filter_images, get_valid_points_indices

class Format(Enum):
    SPARSE = 0
    DENSE = 1



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
    parser.add_argument("-fnd", "--fill_neighbor_distance", type=float, default=1,
                        help="When filling not colored points, this specifies the maximum distance of its neighbors. "
                             "Measured in the unit of measure of the reconstruction/model. Default to 1.")
    parser.add_argument("-ft", "--fill_threshold", type=float, default=0.5,
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
    
    if not args.disable_filling and args.fill_neighbor_distance < 0:
        print("Fill neighbor distance can't be lower than 0.")
        exit()
    if not args.disable_filling and (args.fill_threshold < 0 or args.fill_threshold >= 1): 
        print("Fill threshold must be in [0,1)")
        exit()
    
    if args.image_cap < 0:
        print("Image cap must be at least 0.")
        exit()

    return args

def load_points_and_ids(rec: col.Reconstruction, model_path: str, out_format: Format) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint32] | None]:
    match out_format:
        case Format.SPARSE:
            ids = np.array([point_id for point_id in rec.points3D.keys()], dtype=np.uint32)
            points = np.array([point.xyz for point in rec.points3D.values()], dtype=np.float64)
        case Format.DENSE:
            ply = col.Reconstruction()
            ply.import_PLY(model_path)

            ids = None
            points = np.array([point.xyz for point in ply.points3D.values()], dtype=np.float64)

    if len(points) == 0:
        print(f"No points found in {"reconstruction" if out_format == Format.SPARSE else "model"}.")
        exit()

    return points, ids

def project_colors(image: Image, colors_per_point: list[list[npt.NDArray[np.uint8]]], projected_points: npt.NDArray[np.float64], valid_points_indices: npt.NDArray[np.int32]) -> bool:
    image_bitmap = col.Bitmap.read(image.path, True)
    if image_bitmap is None:
        # could not read image, returning False
        return False
    # the array has shape [height, width, 3]
    image_data = image_bitmap.to_array()
    
    for index, projected_point in zip(valid_points_indices, projected_points[valid_points_indices]):
        # use the truncated projected point coordinates to get the color
        color = image_data[np.floor(projected_point[1]).astype(np.int32), np.floor(projected_point[0]).astype(np.int32)]
        colors_per_point[index].append(color)
    
    return True

def fill_colors(colors: npt.NDArray[np.float64], points: npt.NDArray[np.float64], not_colored_points_indices: npt.NDArray[np.int32], fill_radius: float, fill_threshold: float):
    points_tree = KDTree(points)
    not_colored_points_tree = KDTree(points[not_colored_points_indices])
    neighbors = not_colored_points_tree.query_ball_tree(points_tree, fill_radius)
    for index, neighbors_indices in enumerate(neighbors):
        # if there are no neighbors skip the point
        if not neighbors_indices:
            continue
        
        not_colored_neighbors_indices = np.intersect1d(neighbors_indices, not_colored_points_indices)
        # if the percentage of not colored neighbors is greater than the threshold skip the point
        if len(not_colored_neighbors_indices) / len(neighbors_indices) > fill_threshold:
            continue
        
        # remove not colored neighbors
        neighbors_indices = np.setdiff1d(neighbors_indices, not_colored_neighbors_indices)
        # get the colors of colored neighbors
        colored_neighbor_colors = colors[neighbors_indices]
        
        colors[index] = np.median(colored_neighbor_colors, axis=0)

def write_rec(out_path: str, rec: col.Reconstruction, ids: npt.NDArray[np.uint32], colors: npt.NDArray[np.uint8]):
    for point_id, point_color in zip(ids, colors):
        rec.point3D(point_id).color = point_color
    
    rec.write(out_path)

def write_ply(out_path: str, points: npt.NDArray[np.float64], colors: npt.NDArray[np.uint8]):
    ply = col.Reconstruction()
    dummy_track = col.Track()
    for index in range(points.shape[0]):
        ply.add_point3D(points[index], dummy_track, colors[index])

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
    fill_radius = args.fill_neighbor_distance
    fill_threshold = args.fill_threshold
    image_cap = args.image_cap

    print("Loading data...")
    rec = col.Reconstruction(rec_path)
    points, ids = load_points_and_ids(rec, model_path, out_format)
    images = filter_images(rec, image_folder_path, image_filter)
    
    print("Processing images...")
    colors_per_point = [[] for i in range(points.shape[0])]
    images_num = len(images)
    for index, image in enumerate(images):
        print(f"    Processing {index + 1}th image out of {images_num}...")
        print(f"    Image is {image.path}")

        print("    Projecting points...")
        points_cam_frame = image.cam_from_world(points)
        projected_points = image.project_points_from_cam(points_cam_frame)
        
        print("    Filtering points...")
        valid_indices = get_valid_points_indices(image, points_cam_frame, projected_points, neighbor_distance, max_depth_difference)
        
        print("    Projecting colors...")
        if not project_colors(image, colors_per_point, projected_points, valid_indices):
            print("    Could not load image, skipping it...")
            if image_cap != 0:
                image_cap += 1 # since the image was not processed we need to increase the image cap
            continue

        if image_cap != 0 and index + 1 == image_cap:
            print("Reached image cap.")
            break
        if index + 1 != images_num:
            print("    ---")

    print("Processing projected colors...")
    colors = np.array([np.mean(colors, axis=0)
                            if colors
                            else purple
                            for colors in colors_per_point])
    
    if not disable_filling:
        print("Filling...")
        not_colored_point_indices = np.array([index
                                                for index, colors in enumerate(colors_per_point)
                                                if not colors])
        fill_colors(colors, points, not_colored_point_indices, fill_radius, fill_threshold)

    colors = colors.astype(np.uint8)
    match out_format:
        case Format.SPARSE:
            print(f"Writing reconstruction at {out_path}...")
            write_rec(out_path, rec, ids, colors)
        case Format.DENSE:
            print(f"Writing ply at {out_path}...")
            write_ply(out_path, points, colors)
    print("Finished execution.")

if __name__ == "__main__":
    main()
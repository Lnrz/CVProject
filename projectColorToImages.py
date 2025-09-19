import numpy as np
import numpy.typing as npt
import pycolmap as col
import argparse as argp
from scipy.spatial import KDTree
from common import Image, purple, filter_images, get_valid_points_indices


def get_arguments() -> argp.Namespace:
    """Get the arguments from the command line and check their validity.
    
    If an argument is invalid stop the script.
    """
    parser = argp.ArgumentParser(description="Project Model Color to Images")
    parser.add_argument("reconstruction_path", type=str, help="Path to the reconstruction.")
    parser.add_argument("model_path", type=str, help="Path to the model.")
    parser.add_argument("image_folder_path", type=str, help="Path where the images are stored.")
    parser.add_argument("image_name_filter", type=str, help="Only images with this filter in their names will be used.")
    parser.add_argument("output_path", type=str, help="Path where to write the output.")
    parser.add_argument("-a", "--alpha", type=float, default=1, help="The alpha value of the projected colors. Default to 1.")
    parser.add_argument("-nd", "--neighbor_distance", type=float, default=15,
                        help="When processing a point projected on the image, this specifies the maximum distance of its neighbors. "
                             "Measured in pixel. Default to 15.")
    parser.add_argument("-mdd", "--max_depth_difference", type=float, default=1,
                        help="When processing a point projected on the image, "
                             "this specifies the maximum distance that there can be between the point's distance from the camera "
                             "and the distance from the camera of its neighbor that is nearest to the camera. "
                             "If greater the point will be skipped. Default to 1. "
                             "Measured in the unit of measure of the reconstruction/model")
    parser.add_argument("-ir", "--influence_radius", type=float, default=1,
                        help="Radius used to determine which pixels are affected by a projected point color. "
                             "Measured in pixels. Default to 1.")
    parser.add_argument("-up", "--use_purple", action="store_true", help="Flag to use purple as a color for not colored points instead of leaving the images color as it is.")
    parser.add_argument("-df", "--disable_filling", action="store_true", help="Flag to disable the filling process for not colored pixels.")
    parser.add_argument("-fr", "--fill_radius", type=float, default=5,
                        help="Radius used in the filling process to determine the neighbors of not colored pixels. "
                             "Measured in pixels. Default to 5.")
    parser.add_argument("-ft", "--fill_threshold", type=float, default=0.5,
                        help="The maximum percentage of not colored neighbors a not colored pixel can have to be filled. "
                              "If the percentage is exceeded the pixel is not filled. Default to 0.5.")
    parser.add_argument("-ic", "--image_cap", type=int, default=0, help="Maximum number of images to process. Useful for debugging. 0 means no cap.")
    args = parser.parse_args()
    
    if args.alpha <= 0 or args.alpha > 1:
        print("Alpha must be in (0,1].")
        exit()

    if args.neighbor_distance < 0:
        print("Neighbor distance can't be lower than 0.")
        exit()
    if args.max_depth_difference < 0:
        print("Max depth difference can't be lower than 0.")
        exit()
    if args.influence_radius < 0:
        print("Influence radius can't be lower than 0.")
        exit()

    if not args.disable_filling and args.fill_radius < 0:
        print("Fill radius can't be lower than 0.")
        exit()
    if not args.disable_filling and (args.fill_threshold < 0 or args.fill_threshold >= 1):
        print("Fill threshold must be in [0,1).")
        exit() 

    if args.image_cap < 0:
        print("Image cap must be at least 0.")
        exit()

    return args

def load_points_and_colors(model_path: str) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Load the data from the model found at ``model_path``.
    
    If the model has no points stop the script.

    Return
    - the points, as an numpy array of 3d vectors of ``float64``;
    - the colors, as a numpy array of 3d vectors of ``float64``.
    """
    ply = col.Reconstruction()
    ply.import_PLY(model_path)

    points = np.array([point.xyz for point in ply.points3D.values()], dtype=np.float64)
    colors = np.array([point.color for point in ply.points3D.values()], dtype=np.float64)    
    
    if len(points) == 0:
        print(f"No points found in the model.")
        exit()
    
    return points, colors

def alpha_blend(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64], a: float) -> npt.NDArray[np.uint8]:
    """Alpha blend ``x`` and ``y`` using as x's alpha ``a``."""
    return (a * x + (1 -a) * y).astype(np.uint8)

def project_colors(image: Image, points: npt.NDArray[np.float64], colors: npt.NDArray[np.float64], pixel_tree: KDTree, influence_radius: float, alpha: float, use_purple: bool) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]] | None:
    """Project the colors of ``points`` to ``image``.
    
    The indices of ``points`` and ``colors`` must match, that is: the color of the n-th point of ``points`` is the n-th color of ``colors``.
    
    ``pixel_tree`` is a ``KDTree`` containing the coordinates of the pixels of ``image``.

    ``influence_radius`` specify the influence of a projected point, the color of the point will be added to the list of colors of any pixel under its influence.

    ``alpha`` specify the alpha of the colors projected on the image.

    ``use_purple`` if ``True`` purple will be used for pixels where colors were not projected, otherwise the original pixels'colors will be kept.

    Return:
    - the colors of the new image, as a numpy array of shape ``[height,width,3]`` and type ``float64``
    - and the indices of the pixels that were not colored, as a numpy array where every element specify the coordinates of the pixel as (x,y) and is of type ``int32``.
    - if the projection was not successful, just ``None``.
    """
    # the shape of the array is [height, width, 3], the same col.Bitmap uses
    # so it also means that to access the pixel (x,y) has index [y,x]
    colors_per_pixel = [[[] for x in range(image.width)] for y in range(image.height)]

    image_bitmap = col.Bitmap.read(image.path, True)
    if image_bitmap is None:
        # could not read image, returning None
        return None
    # the array has shape [height, width, 3]
    image_data = image_bitmap.to_array()
    
    for point, color in zip(points, colors):
        affected_pixels_indices = pixel_tree.query_ball_point(point, influence_radius)
        affected_pixels_coordinates = np.floor(pixel_tree.data[affected_pixels_indices]).astype(np.uint32)

        for pixel_coordinates in affected_pixels_coordinates:
            colors_per_pixel[pixel_coordinates[1]][pixel_coordinates[0]].append(color)

    final_colors = np.array([
                                [alpha_blend(np.mean(colors, axis=0), image_data[y, x].astype(np.float64), alpha)
                                    if colors
                                    else (image_data[y, x] if not use_purple 
                                          else alpha_blend(purple, image_data[y, x].astype(np.float64), alpha))
                                for x, colors in enumerate(row)]
                            for y, row in enumerate(colors_per_pixel)])
    not_colored_pixels_coordinates = np.array([(x, y)
                                            for y, row in enumerate(colors_per_pixel)
                                            for x, colors in enumerate(row) if not colors],
                                            dtype=np.uint32)

    return final_colors, not_colored_pixels_coordinates

def fill_colors(final_colors: npt.NDArray[np.float64], not_colored_pixels_coordinates: npt.NDArray[np.uint32], pixel_tree: KDTree, fill_radius: float, fill_threshold: float):
    """Fill ``final_colors`` by taking a median of the neighborhood, if possible, for the pixels specified by ``not_colored_pixels_coordinates``.
    
    ``pixel_tree`` is a ``KDTree`` containing the coordinates of the pixels of ``image``.

    ``fill_radius`` specify the size of the neighborhood, that is what is the maximum distance between two neighbors.
    
    ``fill_threshold`` specify the minimum percentage of colored neighbors a point must have to be filled.
    """
    not_colored_points_indices_set = {(x,y) for x, y in not_colored_pixels_coordinates} # data structure to speed up membership lookup
    for x, y in not_colored_pixels_coordinates:
        neighbors_indices = pixel_tree.query_ball_point((x+0.5, y+0.5), fill_radius)
        if not neighbors_indices:
            continue
        
        # change coordinates from (x+.5,y+.5) to (x,y)
        neighbors_coordinates = np.floor(pixel_tree.data[neighbors_indices]).astype(np.uint32)
        colored_neighbors_coordinates = np.array([coordinates for coordinates in neighbors_coordinates
                                                    if (coordinates[0], coordinates[1]) not in not_colored_points_indices_set]
                                                , dtype=np.uint32)

        # check if there are enough colored neighbors, otherwise skip the pixel
        if len(colored_neighbors_coordinates) / len(neighbors_coordinates) < 1 - fill_threshold:
            continue

        neighbors_colors = np.array([final_colors[coordinate[1], coordinate[0]] for coordinate in colored_neighbors_coordinates])
        final_colors[y, x] = np.median(neighbors_colors, axis=0)

def main():
    args = get_arguments()
    rec_path = args.reconstruction_path
    image_folder_path = args.image_folder_path
    image_filter = args.image_name_filter
    out_path = args.output_path
    model_path = args.model_path
    alpha = args.alpha
    neighbor_distance = args.neighbor_distance
    max_depth_difference = args.max_depth_difference
    influence_radius = args.influence_radius
    use_purple = args.use_purple
    disable_filling = args.disable_filling
    fill_radius = args.fill_radius
    fill_threshold = args.fill_threshold
    image_cap = args.image_cap
    
    print("Loading data...")
    points, colors = load_points_and_colors(model_path)
    images = filter_images(col.Reconstruction(rec_path), image_folder_path, image_filter)
    
    print("Processing images...")
    images_number = len(images)
    for index, image in enumerate(images):
        print(f"    Processing {index + 1}th image out of {images_number}...")
        print(f"    Image is {image.path}")
        
        print("    Projecting points...")
        points_camera_frame = image.cam_from_world(points)
        projected_points = image.project_points_from_cam(points_camera_frame)
        
        print("    Filtering points...")
        valid_indices = get_valid_points_indices(image, points_camera_frame, projected_points, neighbor_distance, max_depth_difference)
        
        print("    Projecting colors...")
        pixel_tree = KDTree(np.array([(i+0.5, j+0.5) for i in range(image.width) for j in range(image.height)], dtype=np.float64))
        final_colors, not_colored_coordinates = project_colors(image, projected_points[valid_indices], colors[valid_indices], pixel_tree, influence_radius, alpha, use_purple)
        if final_colors is None:
            print(f"    Couldn't read image at {image.path}. Skipping it...")
            if image_cap != 0:
                image_cap += 1 # since the image was not processed we need to increase the image cap
            continue

        if not disable_filling:
            print("    Filling...")
            fill_colors(final_colors, not_colored_coordinates, pixel_tree, fill_radius, fill_threshold)
        
        img_out_path = out_path + image.name
        print(f"    Writing image to {img_out_path}...")
        col.Bitmap.from_array(final_colors).write(img_out_path)
        
        if index + 1 == image_cap:
            print("Reached image cap.")
            break
        if index + 1 != images_number:
            print("    ---")
    print("Finished execution.")

if __name__ == "__main__":
    main()
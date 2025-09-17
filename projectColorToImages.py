import numpy as np
import numpy.typing as npt
import pycolmap as col
import argparse as argp
from scipy.spatial import KDTree
from common import Image, filter_images


def get_arguments() -> argp.Namespace:
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

def extract_points_and_colors(model_path: str) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    ply = col.Reconstruction()
    ply.import_PLY(model_path)

    points = np.array([point.xyz for point in ply.points3D.values()], dtype=np.float64)
    colors = np.array([point.color for point in ply.points3D.values()], dtype=np.float64)    
    
    if len(points) == 0:
        print(f"No points found in model.")
        exit()
    
    return points, colors

def alpha_blend(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64], a: float) -> npt.NDArray[np.uint8]:
    return (a * x + (1 -a) * y).astype(np.uint8)

def project_colors(image: Image, points: npt.NDArray[np.float64], colors: npt.NDArray[np.float64], alpha: float, neighbor_radius: float, max_depth_difference: float, use_purple: bool, influence_radius: float, disable_filling: bool, fill_radius: float, fill_threshold: float) -> col.Bitmap | None:
    # the shape of the array is [height, width, 3], the same col.Bitmap uses
    # so it also means that to access the pixel (x,y) the index is [y,x]
    colors_per_pixel = [[[] for i in range(image.width)] for j in range(image.height)]
    # pixel positions are expressed in the (x+.5,y+.5) format
    pixel_positions = np.array([(i+0.5, j+0.5) for i in range(image.width) for j in range(image.height)], dtype=np.float64)
    pixel_tree = KDTree(pixel_positions)

    image_bitmap = col.Bitmap.read(image.path, True)
    if image_bitmap is None:
        print(f"    Couldn't read image at {image.path}. Skipping it...")
        return None
    # the array has shape [height, width, 3]
    image_data = image_bitmap.to_array()

    points_in_cam_frame = image.cam_from_world(points)
    projected_points = image.project_points_from_cam(points_in_cam_frame)
    
    # get valid points i.e. points that:
    #     are not behind the camera
    #     are not outside of the image
    #     do not form an angle with the viewing direction greater than half fov on the x axis
    valid_points_indices = np.array([index for index in range(len(points))
                                        if  not np.isnan(projected_points[index][0]) and
                                            image.is_in_image(projected_points[index]) and
                                            image.is_in_x_field_of_view(points_in_cam_frame[index])]
                                    , dtype=np.int32)
    
    # get the distance from the camera of the valid points
    valid_points_depths = points_in_cam_frame[valid_points_indices][:,2]
    
    projected_points_tree = KDTree(projected_points[valid_points_indices])
    neighbors = projected_points_tree.query_ball_tree(projected_points_tree, neighbor_radius)
    for index, neighbor_indices in enumerate(neighbors):
        lowest_depth = np.min(valid_points_depths[neighbor_indices], initial=np.inf)

        # skip point if in the neighbor there is a point closer to the camera by a certain difference
        if (lowest_depth != np.inf) and (valid_points_depths[index] - lowest_depth > max_depth_difference):
            continue
        
        # get pixels inside the projected point range of influence
        # we can't use "index" to access "projected_points" since it's the index of the point once all non valid points are removed
        # (in "projected_points_tree" we are inserting only the valid points)
        # so we have to first access the "valid_points_indices" array with "index" to get the effective index of the point
        affected_pixels_indices = pixel_tree.query_ball_point(projected_points[valid_points_indices[index]], influence_radius)
        affected_pixels_coordinates = np.floor(pixel_positions[affected_pixels_indices]).astype(np.uint32)

        for pixel_coordinates in affected_pixels_coordinates:
            colors_per_pixel[pixel_coordinates[1]][pixel_coordinates[0]].append(colors[valid_points_indices[index]])

    purple = np.array([178, 0, 254], dtype=np.float64)
    final_colors = np.array([
                                [alpha_blend(np.mean(colors, axis=0), image_data[y, x].astype(np.float64), alpha)
                                    if colors
                                    else (image_data[y, x] if not use_purple 
                                          else alpha_blend(purple, image_data[y, x].astype(np.float64), alpha))
                                for x, colors in enumerate(row)]
                            for y, row in enumerate(colors_per_pixel)])

    if not disable_filling:
        print("    Filling...")
        for i, j in [(i, j) for j, row in enumerate(colors_per_pixel) for i, colors in enumerate(row) if not colors]:
            neighbors_indices = pixel_tree.query_ball_point((i+0.5, j+0.5), fill_radius)
            if not neighbors_indices:
                continue
            
            # change coordinates from (x+.5,y+.5) to (x,y)
            neighbors_coordinates = np.floor(pixel_positions[neighbors_indices]).astype(np.uint32)
            colored_neighbors_coordinates = np.array([coordinates for coordinates in neighbors_coordinates
                                                        if colors_per_pixel[coordinates[1]][coordinates[0]]]
                                                    , dtype=np.uint32)

            # check if there are enough colored neighbors, otherwise skip the pixel
            if len(colored_neighbors_coordinates) / len(neighbors_coordinates) < 1 - fill_threshold:
                continue

            neighbors_colors = np.array([final_colors[coordinate[1], coordinate[0]] for coordinate in colored_neighbors_coordinates], dtype=np.uint8)
            final_colors[j, i] = np.median(neighbors_colors, axis=0).astype(np.uint8)

    return col.Bitmap.from_array(final_colors)

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
    points, colors = extract_points_and_colors(model_path)
    images = filter_images(col.Reconstruction(rec_path), image_folder_path, image_filter)
    
    print("Processing images...")
    images_number = len(images)
    for index, image in enumerate(images):
        print(f"    Processing {index + 1}th image out of {images_number}...")
        print(f"    Image is {image.path}")
        bitmap = project_colors(image, points, colors, alpha, neighbor_distance, max_depth_difference, use_purple, influence_radius, disable_filling, fill_radius, fill_threshold)

        if bitmap is not None:
            img_out_path = out_path + image.name
            print(f"    Writing image to {img_out_path}")
            bitmap.write(img_out_path)

        if index + 1 == image_cap:
            print("Reached image cap.")
            break

if __name__ == "__main__":
    main()
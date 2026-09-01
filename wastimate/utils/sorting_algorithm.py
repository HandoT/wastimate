    # -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 12:30:53 2025

@author: hando
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm
import matplotlib.colors as colors
import imageio


def plot_3d_array_gif(array, cube_size=1, colormap='viridis', alpha_range=(0.2, 1.0), gif_name="rotation.gif", frames=60):
    """Creates a rotating 3D plot of a NumPy array and saves it as a GIF.

    This function visualizes a 3D NumPy array by representing non-zero elements
    as cubes. The color and transparency of each cube are mapped to the
    element's value. The plot is then animated by rotating the view, and the
    frames are saved as a GIF.

    Parameters
    ----------
    array : numpy.ndarray
        The 3D NumPy array to be visualized. Non-zero values are plotted as
        cubes.
    cube_size : int, optional
        The size of each individual cube. The default is 1.
    colormap : str, optional
        The name of the matplotlib colormap to use for coloring the cubes.
        The default is 'viridis'.
    alpha_range : tuple of float, optional
        A tuple (min_alpha, max_alpha) defining the minimum and maximum
        transparency values for the cubes. Transparency is mapped to the
        normalized array values. The default is (0.2, 1.0).
    gif_name : str, optional
        The filename for the output GIF. The default is "rotation.gif".
    frames : int, optional
        The number of frames in the GIF, which determines the smoothness of
        the rotation. The default is 60.

    Notes
    -----
    - This function requires the `matplotlib`, `numpy`, and `imageio` libraries.
    - It saves individual frames to a 'frames' directory before compiling them
      into a GIF.
    - The `draw_cube` function is a nested helper used to render each cube
      within the 3D plot.
    """
    
    def draw_cube(ax, position, size, facecolor, edgecolor):
        """Draws a cube at a given position in 3D space with specified color
        and transparency."""
        x, y, z = position
        vertices = np.array([
            [x, y, z], [x+size, y, z], [x+size, y+size, z], [x, y+size, z],
            [x, y, z+size], [x+size, y, z+size], [x+size, y+size, z+size], [x, y+size, z+size]
        ])
        faces = [[vertices[j] for j in [0,1,2,3]], [vertices[j] for j in [4,5,6,7]], 
                 [vertices[j] for j in [0,1,5,4]], [vertices[j] for j in [2,3,7,6]], 
                 [vertices[j] for j in [0,3,7,4]], [vertices[j] for j in [1,2,6,5]]]
        ax.add_collection3d(Poly3DCollection(faces, facecolors=facecolor, edgecolors=edgecolor))
    
    # Normalize array values for colormap and transparency
    norm = colors.Normalize(vmin=np.min(array[array>0]), vmax=np.max(array))
    cmap = plt.colormaps[colormap]
    alpha_min, alpha_max = alpha_range
    alpha_norm = lambda v: alpha_min + (alpha_max - alpha_min) * norm(v)  
    
    # Create frames
    images = []
    for angle in np.linspace(0, 360, frames):
        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection='3d')
    
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                for k in range(array.shape[2]):
                    if array[i, j, k] != 0:  
                        value = array[i, j, k]
                        color = cmap(norm(value))
                        alpha = alpha_norm(value * 0.3)
                        rgba_color = (*color[:3], alpha)
                        edge_color = (*color[:3], max(alpha, 0.5))
                        draw_cube(ax, (i, j, k), cube_size, rgba_color, edge_color)
    
        # Set limits, labels, and rotation
        ax.set_xlim([0, array.shape[0]])
        ax.set_ylim([0, array.shape[1]])
        ax.set_zlim(array.shape[2], 0)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([array.shape[0], array.shape[1], array.shape[2]])
        ax.view_init(elev=30, azim=angle)
    
        # Colorbar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.15)
        cbar.set_label("Hazard index score")
    
        plt.tight_layout()
        filename = f"frames/frame_{int(angle)}.png"
        plt.savefig(filename, dpi=100)
        images.append(imageio.imread(filename))
        plt.close(fig)
    
    # Save as GIF
    imageio.mimsave(gif_name, images, fps=10)

# Example usage:
# plot_3d_array_gif(your_3d_array)

def plot_3d_array(array, cube_size=1, colormap='viridis', alpha_range=(0.2, 1.0)):
    """Plots a 3D NumPy array as small cubes with colors and transparency based
    on values.

    Parameters
    ----------
    array : numpy.ndarray
        A 3D NumPy array containing the data to be visualized. Non-zero values
        in the array will be represented as cubes in the plot.
    cube_size : float, optional
        The side length of each individual cube.
        The default is 1.
    colormap : str, optional
        The name of the Matplotlib colormap used to color the cubes.
        The default is 'viridis'.
    alpha_range : tuple of float, optional
        A tuple `(min_alpha, max_alpha)` defining the minimum and maximum
        transparency (alpha) values for the cubes. The transparency of each
        cube is scaled proportionally to its normalized value within this range.
        The default is (0.2, 1.0).

    Notes
    -----
    - This function requires the `numpy` and `matplotlib` libraries.
    - It saves the generated 3D plot to a static image file named
      'hazard_index_placement.png' with a resolution of 200 DPI.
    - The plotting is handled by a nested helper function, `draw_cube`, which
      creates the geometric shapes for each data point.

    """
    def draw_cube(ax, position, size, facecolor, edgecolor):
        """Draws a cube at a given position in 3D space with specified color
        and transparency."""
        x, y, z = position
        # Define the vertices of a cube
        vertices = np.array([
            [x, y, z], [x+size, y, z], [x+size, y+size, z], [x, y+size, z],
            [x, y, z+size], [x+size, y, z+size], [x+size, y+size, z+size], [x, y+size, z+size]
        ])
        # Define the six faces
        faces = [[vertices[j] for j in [0,1,2,3]],
                 [vertices[j] for j in [4,5,6,7]], 
                 [vertices[j] for j in [0,1,5,4]], 
                 [vertices[j] for j in [2,3,7,6]], 
                 [vertices[j] for j in [0,3,7,4]], 
                 [vertices[j] for j in [1,2,6,5]]]

        # Add the faces to the plot
        ax.add_collection3d(Poly3DCollection(faces, facecolors=facecolor, edgecolors=edgecolor))

    # Normalize array values for colormap and transparency
    norm = colors.Normalize(vmin=np.min(array[array>0]), vmax=np.max(array))
    cmap = plt.colormaps[colormap]

    # Normalize alpha values between given range
    alpha_min, alpha_max = alpha_range
    alpha_norm = lambda v: alpha_min + (alpha_max - alpha_min) * norm(v)  # Scale alpha

    # Plot setup
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Loop through array and plot cubes where value is nonzero
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            for k in range(array.shape[2]):
                if array[i, j, k] != 0:  # Only plot nonzero elements
                    value = array[i, j, k]
                    color = cmap(norm(value))  # Get color from colormap
                    alpha = alpha_norm(value*0.3)  # Scale alpha
                    rgba_color = (*color[:3], alpha)  # Convert to RGBA with scaled alpha
                    edge_color = (*color[:3], max(alpha, 0.5))  # Slightly dimmed edges
                    
                    draw_cube(ax, (i, j, k), cube_size, rgba_color, edge_color)

    # Set limits and labels
    ax.set_xlim([0, array.shape[0]])
    ax.set_ylim([0, array.shape[1]])
    ax.set_zlim(array.shape[2], 0)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    ax.set_box_aspect([array.shape[0], array.shape[1], array.shape[2]])

    # Add colorbar with spacing
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.15)
    cbar.set_label("Hazard index score")
    plt.tight_layout()

    #plt.show()
    plt.savefig("hazard_index_placement.png", dpi=200)

def edge_distance_order(disposalvalues, shape, valid_mask, mode):
    """Generates a list of 3D coordinates sorted by their combined distance
    from the nearest edge.

    This function prioritizes placement from the bottom up, filling the lowest
    layers of the repository first and then placing any remaining packages in
    the next highest layer, ordered by their distance from the edges of the
    repository. This logic ensures waste is always placed on the ground with
    proper support.

    Parameters
    ----------
    disposalvalues : list or numpy.ndarray
        A 1D array of values representing the items to be placed.
    shape : tuple of int
        The dimensions (x, y, z) of the 3D storage array.
    valid_mask : numpy.ndarray
        A 3D boolean mask of the same shape as the storage array, indicating
        valid (True) or invalid (False) locations for placement.
    mode : str
        The sorting mode.
        - If 'closure', locations are sorted based on their summed edge
          distances.
        - If 'operations', locations are sorted based on their z-coordinate
          (height) and then by their summed edge distances.

    Returns
    -------
    numpy.ndarray
        A 2D array where each row is a `(x, y, z)` coordinate, sorted according
        to the specified distance criteria.

    Notes
    -----
    The distance metric is calculated as the sum of the square roots of the
    distances to the nearest edges along each dimension. The sorting is done in
    descending order of these distances, ensuring that locations further from
    the edges are filled first.
    """

    dims = len(shape)
    counts = len(disposalvalues)
    layers_needed = np.ceil(counts / (shape[0] * shape[1]))
    free_packages = counts - int((shape[0] * shape[1]) * (layers_needed - 1))
    
    # Generate all (x, y, z) indices
    indices = np.array(np.meshgrid(np.arange(shape[0]),np.arange(shape[1]),np.arange(shape[2]),indexing='ij')).reshape(dims, -1).T

    # Keep only indices where valid_mask is True
    valid_bot_indices = indices[(valid_mask[indices[:, 0], indices[:, 1], indices[:, 2]]) & (indices[:, 2] >= shape[2] - layers_needed + 1)]
    valid_top_indices = indices[(valid_mask[indices[:, 0], indices[:, 1], indices[:, 2]]) & (indices[:, 2] == shape[2] - layers_needed)]

    top_distances = np.min(np.stack([valid_top_indices, np.array(shape) - 1 - valid_top_indices], axis=2), axis=2)
    top_summed_distances = np.sum(top_distances**0.5, axis=1)
    sorted_top_idx = np.argsort(top_summed_distances)[::-1][:free_packages]  # Indices of sorted 1D array in descending order

    total_indices = np.concatenate((valid_bot_indices, valid_top_indices[sorted_top_idx]))
    total_distances = np.min(np.stack([total_indices, np.array(shape) - 1 - total_indices], axis=2), axis=2)
    
    if mode == "closure":
        total_summed_distances = np.sum(total_distances**0.5, axis=1)
        
    elif mode == "operations":
        total_summed_distances = (total_indices[:,-1]) + np.sum(total_distances**0.5, axis=1)
    
    
    sorted_total_idx = np.argsort(total_summed_distances)[::-1]  # Indices of sorted 1D array in descending order

    # Create an empty list to hold the sorted 2D array
    sorted_valid_indices = np.empty_like(total_indices)
    for i, idx in enumerate(sorted_total_idx):
        sorted_valid_indices[i] = total_indices[idx]

    #print(sorted_indices)
    return sorted_valid_indices  # Return sorted coordinate list

def replace_values_with_indices(three_d_array, one_d_array):
    """Replaces values in a 3D array with their corresponding indices from a 1D
    array.

    This function creates a mapping from each unique value in `one_d_array` to
    its index. It then applies this mapping to the `three_d_array`, replacing
    each value with its corresponding index. NaN values are preserved.

    Parameters
    ----------
    three_d_array : numpy.ndarray
        The 3D array whose values are to be replaced.
    one_d_array : numpy.ndarray
        The 1D array that provides the mapping from value to index.

    Returns
    -------
    numpy.ndarray
        A new 3D array of the same shape as `three_d_array`, where each value
        has been replaced by its index from `one_d_array`. NaN values
        in the original array remain NaN.
    """

    # Create a mapping from value -> index
    value_to_index = {val: idx for idx, val in enumerate(one_d_array)}

    # Create an integer array filled with -1 (default for missing values)
    index_array = np.full(three_d_array.shape, -1, dtype=float)  # Use float to allow NaN

    # Mask for NaN values in the original 3D array
    nan_mask = np.isnan(three_d_array)

    # Apply index mapping where values match
    for val, idx in value_to_index.items():
        index_array[three_d_array == val] = idx

    # Preserve NaNs
    index_array[nan_mask] = np.nan

    return index_array

def fill_config(disposalvalues, storagearray, mode="closure"):
    """Fills a 3D storage array with sorted values from the edges inward.

    This function first sorts the input `disposalvalues` in descending order.
    It then uses the `edge_distance_order` helper function to determine the
    optimal placement coordinates in the 3D storage array, prioritizing
    locations farthest from the edges. The largest values from `disposalvalues`
    are placed at these optimal locations. Finally, it generates a second array
    where the values are replaced by their original indices.

    Parameters
    ----------
    disposalvalues : list or numpy.ndarray
        A 1D array of values (e.g., hazard index scores) to be placed.
    storagearray : numpy.ndarray
        An empty or pre-filled 3D NumPy array representing the storage space.
        It should have the same shape as the desired output arrays.
    mode : str, optional
        The sorting mode for the placement coordinates. Can be 'closure' or
        'operations'. The default is 'closure'.

    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing two 3D arrays:
        - The `EmptyRepository` array, filled with the sorted `disposalvalues`.
        - The `IndexedRepository` array, with the values replaced by their
          original indices from the `disposalvalues` list.
    """

    EmptyRepository = np.zeros_like(storagearray)

    # Create a mask for valid locations (where PackageConfiguration is not NaN or None)
    valid_mask = ~np.isnan(storagearray)

    # Flatten and sort non-zero elements from largest to smallest
    flattened_config = np.array(disposalvalues)
    non_zero_elements = np.sort(flattened_config[flattened_config > 0.0])[::-1]

    # Get valid coordinate order based on edge distance
    coords = edge_distance_order(disposalvalues, EmptyRepository.shape, valid_mask, mode)

    # Fill in elements in this order
    element_idx = 0
    for x, y, z in coords:
        if element_idx >= len(non_zero_elements):  # Stop if we run out of elements
            break
        EmptyRepository[x, y, z] = non_zero_elements[element_idx]
        element_idx += 1
    
    IndexedRepository = replace_values_with_indices(EmptyRepository, flattened_config)

    return EmptyRepository, IndexedRepository

def group_sorted_values_with_indices(values, N):
    """Sorts a list of values and groups them into sublists of size N,
    while also preserving and grouping their original indices.

    This function is useful for batching operations or partitioning a dataset
    after sorting. It maintains the link between the original positions of the
    values and their new, sorted positions within the groups.

    Parameters
    ----------
    values : list
        A list of numeric values to be sorted and grouped.
    N : int
        The desired size of each subgroup.

    Returns
    -------
    tuple of list of list
        A tuple containing two lists of lists:
        - `grouped_values`: A list of sublists, where each sublist contains
          `N` sorted values from the original list.
        - `grouped_indices`: A list of sublists, where each sublist contains
          the original indices of the values in the corresponding
          `grouped_values` sublist.
    """

    # Pair values with their original indices
    indexed_values = list(enumerate(values))  # [(index, value), ...]
    
    # Sort by value
    indexed_values.sort(key=lambda x: x[1])  # Sort by the second element (value)
    
    # Extract sorted values and indices
    sorted_values = [v for _, v in indexed_values]
    sorted_indices = [i for i, _ in indexed_values]

    # Group values and indices into sublists of size N
    grouped_values = [sorted_values[i:i + N] for i in range(0, len(sorted_values), N)]
    grouped_indices = [sorted_indices[i:i + N] for i in range(0, len(sorted_indices), N)]

    return grouped_values, grouped_indices

            
def swap_elements_old(list_of_lists, problematic_indices, occurrences):
    """Attempts to swap problematic elements with smaller, unproblematic
    elements.

    This function is an iterative optimization algorithm that tries to find a
    new, valid state for a configuration by swapping "problematic" values (the
    largest values in a given sublist) with smaller, available values. The 
    process continues until a unique configuration is found that has not been
    seen before.

    Parameters
    ----------
    list_of_lists : list of list
        The current configuration of values, represented as a list of sublists.
    problematic_indices : list of int
        A list of indices corresponding to the sublists that contain the values
        to be swapped.
    occurrences : list of tuple
        A list of previously seen configurations, used to check for uniqueness.

    Returns
    -------
    tuple of (list of list, list)
        - The `list_of_lists` with elements swapped to a new, unique state.
        - The `occurrences` list, updated with the new state.
        If no unique solution is found after 10,000 attempts, the original
        list and an empty list are returned.
    """

    def find_index_of_number(list_of_lists, number):
        for list_index, sublist in enumerate(list_of_lists):
            if number in sublist:
                element_index = sublist.index(number)
                return list_index, element_index
    
    def transmute_list(list_of_lists, problem_values):        
        swapped_values = []
        for val in problem_values:
            pl_idx, pe_idx = find_index_of_number(list_of_lists, val)
            test_val = val
            while test_val > 0:
                test_val -= 1
                if test_val not in problem_values and test_val not in swapped_values:
                    ridx1, cidx1 = find_index_of_number(list_of_lists, test_val)
                    ridx2, cidx2 = find_index_of_number(list_of_lists, val)
                    list_of_lists[ridx1][cidx1], list_of_lists[ridx2][cidx2] = list_of_lists[ridx2][cidx2], list_of_lists[ridx1][cidx1]
                    
                    swapped_values.append(test_val)
                    break
                
        return list_of_lists, swapped_values

    TempListOfLists = [sublist.copy() for sublist in list_of_lists]
    problem_values = []
    for pidx in problematic_indices:
        problem_list = list_of_lists[pidx]
        problem_values.append(max(problem_list))
    
    loop = 10000
    while True:
        loop -= 1
        list_of_lists, swapped_values = transmute_list(list_of_lists, problem_values)
        
        TempState = tuple(frozenset(tuple(sorted(sublist))) for sublist in list_of_lists)
        if TempState not in occurrences:
            occurrences.append(TempState)
            return list_of_lists, occurrences
        else:
            problem_values = swapped_values
        
        if loop < 0:
            print("No optimal solutions could be found.")
            break
    
    return TempListOfLists, []


def storage_to_excel(list1, list2, list3=None, error_list=None, filename="storage_output.xlsx"):
    """Saves two 3D lists (lists of lists of lists) into an Excel file.

    The function takes two 3D lists and writes them side-by-side into a single
    Excel file. It flattens the 3D lists into 2D sections, with a blank row
    separating each layer from the original 3D structure. The function can also
    add a special column to highlight rows identified by `error_list`. Layers
    that contain only empty strings are skipped to maintain a clean output.

    Parameters
    ----------
    list1 : list of list of list
        The first 3D list of data to be saved.
    list2 : list of list of list
        The second 3D list of data to be saved.
    error_list : list of int, optional
        A list of row indices to be marked with "NB!". This is for indicating
        problematic packages or locations. The default is None.
    filename : str, optional
        The name of the Excel file to save the data to.
        The default is "storage_output.xlsx".

    Returns
    -------
    None
        The function does not return any value but saves the data to an Excel
        file.
    """

    # Flatten the 3D lists into multiple 2D sections
    if list3 is None:
        data1, data2 = [], []
        
        for layer1, layer2 in zip(list1, list2):
            # Check if layer1 or layer2 contains only empty strings
            if not all(cell == "" for row in layer1 for cell in row):
                data1.extend(layer1 + [[""] * len(layer1[0])])  # Add row gap if it's not all empty
            if not all(cell == "" for row in layer2 for cell in row):
                data2.extend(layer2 + [[""] * len(layer2[0])])  # Add row gap if it's not all empty
        
        # Convert to DataFrame
        
        if error_list is not None and len(error_list) > 0:
            data0 = [[""] for _ in range(len(data1))]
            for index in error_list:
                data0[index] = ["Does not satisfy criteria!"]
            df0 = pd.DataFrame(data0)
        df1 = pd.DataFrame(data1)
        df2 = pd.DataFrame(data2)
        
        # Concatenate side by side with a column gap
        if error_list is not None and len(error_list) > 0:
            df_combined = pd.concat([df0, pd.DataFrame([""] * len(df1)), df1, pd.DataFrame([""] * len(df1)), df2], axis=1)
    
        else:
            df_combined = pd.concat([df1, pd.DataFrame([""] * len(df1)), df2], axis=1)
    else:
        data1, data2, data3 = [], [], []
        
        for layer1, layer2, layer3 in zip(list1, list2, list3):
            # Check if layer1 or layer2 contains only empty strings
            if not all(cell == "" for row in layer1 for cell in row):
                data1.extend(layer1 + [[""] * len(layer1[0])])  # Add row gap if it's not all empty
            if not all(cell == "" for row in layer2 for cell in row):
                data2.extend(layer2 + [[""] * len(layer2[0])])  # Add row gap if it's not all empty
            if not all(cell == "" for row in layer3 for cell in row):
                data3.extend(layer3 + [[""] * len(layer3[0])])  # Add row gap if it's not all empty
        
        # Convert to DataFrame
        
        if error_list is not None and len(error_list) > 0:
            data0 = [[""] for _ in range(len(data1))]
            for index in error_list:
                data0[index] = ["Does not satisfy criteria!"]
            df0 = pd.DataFrame(data0)
        df1 = pd.DataFrame(data1)
        df2 = pd.DataFrame(data2)
        df3 = pd.DataFrame(data3)
        
        # Concatenate side by side with a column gap
        if error_list is not None and len(error_list) > 0:
            df_combined = pd.concat([df0, pd.DataFrame([""] * len(df1)), df1, pd.DataFrame([""] * len(df1)), df2, pd.DataFrame([""] * len(df2)), df3], axis=1)
    
        else:
            df_combined = pd.concat([df1, pd.DataFrame([""] * len(df1)), df2, df2, pd.DataFrame([""] * len(df2)), df3], axis=1)
    

    # Save to Excel
    df_combined.to_excel(filename, index=False, header=False)

# Example usage                
if __name__ == "__main__":
    # Example input: lists with some problematic indices (lists that exceed a certain limit)
    list_of_lists = [[1, 2, 5], [3, 4, 7], [8, 6, 9]]
    problematic_indices = [0, 1]  # Let's say list 0 and list 1 are problematic
    
    #updated_lists = swap_elements(list_of_lists, problematic_indices)
    #print(updated_lists)
    
    
    # StorageArray = np.zeros((6,6,9))
    # DisposalValues = np.linspace(0.1, 10, 100)
    # filled_repo, _ = fill_config(DisposalValues, StorageArray)
    
    # #best_state = strategic_positioning(DisposalArray)
    # plot_3d_array_gif(filled_repo)
    # #plot_3d_array(filled_repo, colormap='coolwarm')  # Call function with custom colormap
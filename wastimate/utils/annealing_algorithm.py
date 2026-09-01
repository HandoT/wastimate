import numpy as np

def initialize_groups(values, N):
    """Initializes groups of size N and one smaller group if necessary.

    This function takes a 1D list or NumPy array of values, shuffles them
    randomly, and then partitions them into groups of a specified size. If the
    total number of values is not perfectly divisible by N, the last group will
    contain the remainder.

    Parameters
    ----------
    values : list or numpy.ndarray
        A list of numeric values to be grouped.
    N : int
        The desired size for each group.

    Returns
    -------
    list of list
        A list of lists, where each inner list is a group of values. All but
        the last group will be of size N.
    """
    
    M = len(values)
    
    # If there are fewer samples than group size, just put them all in one group
    if M <= N:
        return [values]
    
    num_groups = M // N
    remainder = M % N
    
    np.random.shuffle(values)  # Shuffle for randomness
    groups = [values[i * N: (i + 1) * N] for i in range(num_groups)]
    
    # Handle remainder if it exists
    if remainder > 0:
        groups.append(values[num_groups * N:])
    
    return groups

def compute_cost(groups, error_idx):
    """
    Computes a cost based on the sum of values in "disqualified" groups.

    This function calculates a cost for a given grouping of values. The cost is
    a penalty applied to groups identified by `error_idx`. The penalty is higher
    for groups with smaller sums, which encourages the optimization algorithm
    to move large values out of these problematic groups. If there are no
    disqualified groups, the cost is zero.

    Parameters
    ----------
    groups : list of list
        A list of lists representing the current grouping of values.
    error_idx : list of int
        A list of indices corresponding to the "disqualified" groups.

    Returns
    -------
    float
        The calculated cost.
    """

    # Primary term: Number of disqualified lists (minimize this)
    if len(error_idx) < 1:
        return 0
    else:
        cost = np.sum([1000/(sum(groups[idx])+1e-3) for idx in error_idx])
        return cost

def swap_elements(groups, error_idx):
    """
    Swaps an element from an error group with an element from a random group.

    This function performs a single step in a local search or simulated
    annealing algorithm. It selects a random group from the `error_idx` list
    and identifies its largest element. It then selects another random group
    and swaps this largest "problematic" element with a random element from the
    second group. This aims to move high-value items out of groups that are 
    causing a high cost.

    Parameters
    ----------
    groups : list of list
        The current list of grouped values.
    error_idx : list of int
        A list of indices of the groups to be considered for a swap.

    Returns
    -------
    list of list
        A new list of lists with the elements swapped. The original lists are
        not modified.
    """
    
    new_groups = [list(group) for group in groups]  # Deep copy
    
    # If there are fewer than two groups, no swap can occur
    if len(groups) < 2 or len(error_idx) < 1:
        return new_groups
    
    group_a = np.random.choice(error_idx)
    group_b = np.random.choice(len(groups))

    # Avoid empty groups
    if not new_groups[group_a] or not new_groups[group_b]:
        return new_groups

    
    idx_a = np.argmax(new_groups[group_a])#np.random.choice(len(new_groups[group_a]))
    idx_b = np.random.choice(len(new_groups[group_b]))

    # Swap elements
    new_groups[group_a][idx_a], new_groups[group_b][idx_b] = (
        new_groups[group_b][idx_b], new_groups[group_a][idx_a]
    )
    return new_groups

def simulated_annealing(values, error_idx, max_iterations=10000, initial_temp=10, cooling_rate=0.995):
    """
    Performs a simulated annealing optimization to find an optimal grouping.

    This function applies the simulated annealing algorithm to solve a grouping
    problem. Starting with an initial state, it iteratively generates new 
    states by swapping elements (`swap_elements`) and evaluates them based on a
    cost function (`compute_cost`). It accepts better solutions and
    probabilistically accepts worse ones to escape local minima. As the
    temperature cools, the probability of accepting worse solutions decreases.

    Parameters
    ----------
    values : list of list
        The initial grouping of values.
    error_idx : list of int
        A list of indices of groups that are considered "problematic" or
        "disqualified".
    max_iterations : int, optional
        The maximum number of iterations to run the algorithm.
        The default is 10000.
    initial_temp : float, optional
        The starting temperature for the annealing process. A higher temperature
        allows for more exploration. The default is 10.
    cooling_rate : float, optional
        The rate at which the temperature decreases. Must be between 0 and 1.
        A value closer to 1 results in slower cooling. The default is 0.995.

    Returns
    -------
    list of list
        The best grouping of values found during the optimization process.
    """
    
    current_state = [group.copy() for group in values]
    current_cost = compute_cost(current_state)
    best_state, best_cost = current_state, current_cost
    temp = initial_temp

    for _ in range(max_iterations):
        new_state = swap_elements(current_state, error_idx)
        new_cost = compute_cost(new_state)
        cost_diff = new_cost - current_cost

        # Accept better solution or probabilistically accept worse ones
        if cost_diff < 0 or np.random.rand() < np.exp(-cost_diff / temp):
            current_state, current_cost = new_state, new_cost
            if new_cost < best_cost:
                best_state, best_cost = new_state, new_cost

        # Decrease temperature
        temp *= cooling_rate
        if temp < 1e-6:
            break

    return best_state

def reorder_list(original, sorted_original, target):
    """Reorders one list based on the sorted order of another list.

    This function aligns the elements of the `target` list to match the sorted
    order of the `original` list. It first creates a mapping from each element
    in the `original` list to its initial index. Then, it uses this mapping
    to find the correct new positions for the elements of the `target` list.
    This works for both 1D lists of single values and 2D lists of tuples.

    Parameters
    ----------
    original : list
        The list whose sorted order dictates the new order. Can be 1D or 2D.
    sorted_original : list
        The sorted version of the `original` list.
    target : list
        The list to be reordered.

    Returns
    -------
    list
        A new list with elements from `target` reordered to match the sorted
        order of `original`.

    Raises
    ------
    ValueError
        If the input lists do not have the same length.
    """
    
    if len(original) != len(sorted_original) or len(original) != len(target):
        raise ValueError("All lists must have the same length")

    # If elements are tuples (2D data), create an index map based on the tuple's content
    if isinstance(original[0], tuple):
        # Treat tuples as composite elements for sorting
        index_map = {tuple(val): i for i, val in enumerate(original)}
    else:
        # Handle the case for single values (1D data)
        index_map = {val: i for i, val in enumerate(original)}

    # Generate the sorted indices based on the sorted order of original elements
    sorted_indices = [index_map[tuple(val) if isinstance(val, tuple) else val] for val in sorted_original]

    return [target[i] for i in sorted_indices]

if __name__ == "__main__":
    # Example values (2D list)
    error_idx = [3]
    values = [[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13]]
    
    # Running the simulated annealing algorithm
    best_state = simulated_annealing(values, error_idx)
    
    # Printing the results
    print("Best state:")
    for group in best_state:
        print(group)
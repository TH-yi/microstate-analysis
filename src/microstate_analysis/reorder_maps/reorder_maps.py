import numpy as np

def reorder_maps(data, order, one_task=False):
    """
    Reorder the 'maps' in the input data according to the provided indices.

    Example 1:
        Input data:
        data = {
            "task1": {"maps": [[1]*63, [2]*63, [3]*63, [4]*63, [5]*63, [6]*63]},
            "task2": {"maps": [[6]*63, [5]*63, [4]*63, [3]*63, [2]*63, [1]*63]},
        }

        Order indices:
        order = {
            "task1": [3, 5, 4, 1, 0, 2],
            "task2": [5, 4, 3, 2, 1, 0]
        }

        Result after calling the function:
        {
            "task1": {"maps": [[4]*63, [6]*63, [5]*63, [2]*63, [1]*63, [3]*63]},
            "task2": {"maps": [[1]*63, [2]*63, [3]*63, [4]*63, [5]*63, [6]*63]},
        }

    Example 2:
        Input data:
        data = {"maps": [[0]*63, [1]*63, [2]*63, [3]*63, [4]*63, [5]*63]}
        order = [5, 1, 4, 0, 2, 3]

        Result:
        {"maps": [[5]*63, [1]*63, [4]*63, [0]*63, [2]*63, [3]*63]}
    """
    reordered_data = {}
    if one_task:
        # Extract maps and convert to a numpy array for reordering
        task_maps = data['maps']
        if isinstance(task_maps, list):
            task_maps = np.array(task_maps)

        # Perform reordering
        reordered_maps = task_maps[order]
        reordered_data = {"maps": reordered_maps.tolist()}  # Convert back to list format
    else:
        for task_name, indices in order.items():
            if task_name not in data:
                raise ValueError(f"Task '{task_name}' not found in the input data.")

            # Extract maps and convert to a numpy array for reordering
            task_maps = data[task_name]['maps']
            if isinstance(task_maps, list):
                task_maps = np.array(task_maps)

            # Validate that the order indices match the length of maps
            if len(task_maps) != len(indices):
                raise ValueError(f"Indices length mismatch for task '{task_name}'.")

            # Perform reordering
            reordered_maps = task_maps[indices]
            reordered_data[task_name] = {"maps": reordered_maps.tolist()}  # Convert back to list format

    return reordered_data

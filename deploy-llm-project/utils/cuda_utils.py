from torch import cuda as torch_cuda


def get_gpu_memory() -> int:
    """
    Returns the amount of free memory in MB for each GPU.
    """
    return int(torch_cuda.mem_get_info()[0] / (1024**2))


def calculate_layer_count() -> int | None:
    """
    Calculates the number of layers that can be used on the GPU.
    """
    is_gpu_enabled = torch_cuda.is_available()
    if not is_gpu_enabled:
        return None
    LAYER_SIZE_MB = (
        120.6  # This is the size of a single layer on VRAM, and is an approximation.
    )
    # The current set value is for 7B models. For other models, this value should be changed.
    LAYERS_TO_REDUCE = 6  # About 700 MB is needed for the LLM to run, so we reduce the layer count by 6 to be safe.
    if (get_gpu_memory() // LAYER_SIZE_MB) - LAYERS_TO_REDUCE > 32:
        return (get_gpu_memory() // LAYER_SIZE_MB) - LAYERS_TO_REDUCE
    return None

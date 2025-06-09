import time
import tracemalloc
from rich.console import Console
from rich.table import Table

# Display GPU free memory

def free_mem():
    import pynvml
    from rich.console import Console
    console = Console()    
    pynvml.nvmlInit()
    handle=pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    free_amt = info.free / 1024**3
    if free_amt < 4:
        console.print(f"Free memory: [bright_red]{free_amt:.2f} GB[/]")
    else:
        console.print(f"Free memory: [bright_green]{free_amt:.2f} GB[/]")
    pynvml.nvmlShutdown()
    return info.free / 1024**3

# Display GPU memory usage

def profile_function(func):
    import pynvml
    def wrapper(*args, **kwargs):

        # Retrieve verbosity from kwargs or use the default
        verbose = kwargs.pop('verbose', True)

        def start_track_duration():
            return time.time()

        def end_track_duration(start_duration):
            end_time = time.time()
            return end_time - start_duration

        def get_cpu_memory():
            stats = tracemalloc.take_snapshot().statistics('lineno')
            memory = sum(stat.size for stat in stats)
            return memory

        def start_track_cpu_memory():
            tracemalloc.start()
            return get_cpu_memory()
        
        def end_track_cpu_memory(start_memory):
            end_memory = get_cpu_memory()
            tracemalloc.stop()
            memory_diff = end_memory - start_memory
            return memory_diff, end_memory
        
        # Function to track CUDA memory usage
        def start_track_cuda_memory():
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            init_cuda_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return init_cuda_info.free
        
        def end_track_cuda_memory(init_free_cuda_mem):
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)            
            final_cuda_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            final_free_cuda_mem = final_cuda_info.free
            cuda_memory_diff = init_free_cuda_mem - final_cuda_info.free
            return cuda_memory_diff, final_free_cuda_mem        

        # statistics before
        start_duration = start_track_duration()
        start_memory = start_track_cpu_memory()
        start_free_cuda_mem = start_track_cuda_memory()

        result = func(*args, **kwargs)

        # statistics after
        duration = end_track_duration(start_duration)
        memory_diff, end_memory = end_track_cpu_memory(start_memory)
        cuda_memory_diff, final_free_cuda_mem = end_track_cuda_memory(start_free_cuda_mem)

        # Report using rich
        if verbose:
            console = Console()
            table = Table(title=f"Performance {func.__name__}", show_header=True, header_style="bold purple")
            table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
            table.add_column("Initial Value", justify="right", style="green")
            table.add_column("Change Value", justify="right", style="yellow")
            table.add_column("Final Value", justify="right", style="magenta")

            table.add_row("Function Duration", "-", f"{duration:.2f} seconds", f"{duration:.2f} seconds")
            table.add_row("CPU Memory", f"- MB", f"{memory_diff / 1024 ** 2:.2f} MB", f"{end_memory / 1024 ** 2:.2f} MB")
            table.add_row("CUDA Memory", f"{start_free_cuda_mem / 1024**3:.2f} GB", f"{cuda_memory_diff / 1024**3:.2f} GB", f"{final_free_cuda_mem / 1024**3:.2f} GB")

            console.print(table)

        return result

    return wrapper
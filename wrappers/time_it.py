import time

def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        time_diff = end_time - start_time
        hours =  time_diff // 3600
        time_diff %= 3600
        minutes = time_diff // 60
        seconds = time_diff % 60
        print(f"{func.__name__} took {hours} hours, {minutes} minutes, and {seconds:.1f} seconds to execute.")
        return result
    return wrapper
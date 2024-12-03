import random


def utils_list_sample(arr, k):
    if len(arr) > k:
        arr = random.sample(arr, k)
    return arr

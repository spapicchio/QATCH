from __future__ import annotations

import random


def utils_list_sample(arr: list[str], k: int, val: str | None = None):
    """
    Returns a sampling list from the input array, allows for forced inclusion of a value.

    This function takes as input an array, a desired sample size, and an optional value to include in the sample.
    If the array length exceeds the desired sample size, a random sample of size k is created.
    If a value is provided and it exists in the array but not in the created sample,
    it is forcibly added to the sampling by replacing the first element.
    Otherwise, if the array's size is less or equal than k, it returns the whole array.

    Args:
        arr (list): The input array to sample from.
        k (int): The desired sample size.
        val (str | None): The value to be forcibly included in the sample if it exists in the array. Defaults to None.

    Returns:
        list: A sample list from the input array.

    Note:
        - `val` takes any data type that exists in the array.
        - If `val` exists in the array but not in the sample, it replaces the first element in the sampled list.
    """
    if len(arr) > k:
        sampled_arr = random.sample(arr, k)
        if val is not None and utils_check_in_arr(val, sampled_arr) is None and utils_check_in_arr(val, arr):
            sampled_arr[0] = utils_check_in_arr(val, arr)
    else:
        sampled_arr = arr
    return sampled_arr


def utils_check_in_arr(val, arr):
    """
    Check if a value exists in an array and returns the matched value from the array.

    This function takes as input an array and a value. It traverses the array and compares each array value with the
    provided value following a case-insensitive approach. If a match is found, it returns the matched value from
    the array else it returns None.

    Args:
        val (str): The value to be searched in the array.
        arr (list): The input array in which the value will be searched.

    Returns:
        str | None: The matched value from the array if found else None.

    Note:
        - `val` is searched in a case-insensitive manner in the array.
        - If multiple copies of `val` exist in the array, it simply returns the first matched value.
    """

    for arr_val in arr:
        if arr_val.lower() == val.lower():
            return arr_val

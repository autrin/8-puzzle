import re


def get_puzzle():
    path = input("Enter the file path: ")
    try:
        inFile = open(path, "r")
    except FileNotFoundError:
        print("\nFile not found!\n")
        return None
    puzzle = []
    for line in inFile:
        puzzle = puzzle + line.split(" ")
    inFile.close()
    puzzle = [int(x.replace("\n", "").replace("_", "0")) for x in puzzle]
    return puzzle


def ask_algo():
    while True:
        algo = input(
            "Which one of these algorithms would you like to try?\n'BFS', 'IDS', 'h1', 'h2', 'h3': "
        )
        if algo in ["BFS", "IDS", "h1", "h2", "h3"]:
            return algo
        else:
            print(
                "Invalid response! Please choose among 'BFS', 'IDS', 'h1', 'h2', 'h3'."
            )


def count_split_inversion(b1, b2):
    i, j, inv_count = 0, 0, 0
    merged_list = []
    while i < len(b1) and j < len(b2):
        if b1[i] <= b2[j]:
            merged_list.append(b1[i])
            i += 1
        else:
            merged_list.append(b2[j])
            inv_count += len(b1) - i
            j += 1
    merged_list += b1[i:] + b2[j:]  # Add remaining elements
    return merged_list, inv_count


def count_rank_inversion(arr):
    if len(arr) <= 1:
        return arr, 0
    else:
        mid = len(arr) // 2
        left, inv_left = count_rank_inversion(arr[:mid])
        right, inv_right = count_rank_inversion(arr[mid:])
        merged, inv_split = count_split_inversion(left, right)
        total_inversions = inv_left + inv_right + inv_split
        return merged, total_inversions


def check_solvable(puzzle):
    # Exclude the empty tile (0) before calculating inversions
    puzzle_without_empty = [num for num in puzzle if num != 0]
    _, total_inversions = count_rank_inversion(puzzle_without_empty)
    print(f"The number of inversion pairs is: {total_inversions}")
    if total_inversions % 2 == 0:
        return True
    else:
        return False


# Testing
puzzle = get_puzzle()
print(puzzle)

if puzzle is not None:
    if check_solvable(puzzle):
        print("The puzzle is solvable.")
    else:
        print("The puzzle is not solvable.")
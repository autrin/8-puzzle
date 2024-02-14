# scan the file path, and return the array representation of the puzzle.
def get_puzzle():
    path = input("Enter the file path: ")
    try:
        inFile = open(path, 'r')
    except FileNotFoundError:
        print("\nFile not found!\n")
        return None
    puzzle = []
    for line in inFile:
        puzzle = puzzle + line.split(" ") # TODO: there is a '_' in the array and so not every element is an integer
    inFile.close()
    return puzzle

def ask_algo():
    algo = input(
        "Which one of these algorithms would you like to try?\n'BFS', 'IDS', 'h1', 'h2', 'h3'"
    )
    if algo not in ["BFS", "IDS", "h1", "h2", "h3"]:
        print("Invalid response! Write")
    else:
        return True

def count_split_inversion(b1, b2):
    i, j, k, c3 = 0
    b = []
    while i < len(b1) and j < len(b2):
        if b1[i] < b2[j]:  # No split inversion
            b[k] = b[i]
            i = i + 1
            k = k + 1
        elif b1[i] > b2[j]:  # Split inversion
            c3 = c3 + len(b1) - i + 1
            b[k] = b2[j]
            j = j + 1
            k = k + 1
    return (b, c3)

def count_rank_inversion(arr): # TODO: there is a '_' in the array and so not every element is an integer
    a1 = arr[:len(arr//2)]  # left half side of the puzzle
    a2 = arr[len(arr)//2:]  # right half side of the puzzle
    (b1, c1) = count_rank_inversion(a1)
    (b2, c2) = count_rank_inversion(a2)
    (b, c3) = count_split_inversion(b1, b2)
    return (b, c1 + c2 + c3)

# Check if the puzzle is solvable. If a puzzle is solvable then it has even number of inversion pairs.
# Otherwise it is not solvable.
def check_solvable(puzzle):
    (b, c) = count_rank_inversion(puzzle)
    if c % 2 == 0:
        return True
    else:
        return False

# Testing    
puzzle  = get_puzzle()
print(check_solvable(puzzle))
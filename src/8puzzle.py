import argparse
import heapq
import math
import queue
import random
import time
import sys
import os
from collections import defaultdict, deque, Counter
from multiprocessing import Process, Queue

from numpy import mean


class Problem(object):

    def __init__(self, initial=None, goal=None,node_counter=0, **kwds): 
        self.initial = initial
        self.goal = goal
        self.node_counter = 0  # Explicitly initialize node_counter here
        for key, value in kwds.items():
            setattr(self, key, value)

    def actions(self, state):        raise NotImplementedError
    def result(self, state, action): raise NotImplementedError
    def is_goal(self, state):        return state == self.goal
    def action_cost(self, s, a, s1): return 1 # Assuming the cost is 1 for every action/move
    def h(self, node):               return 0
    
    def __str__(self):
        return '{}({!r}, {!r})'.format(
            type(self).__name__, self.initial, self.goal)
    
class Node:
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost

    def __repr__(self): return '<{}>'.format(self.state)
    def __len__(self): return 0 if self.parent is None else (1 + len(self.parent))
    def __lt__(self, other): return self.path_cost < other.path_cost
    
    
failure = Node('failure', path_cost=math.inf) # Indicates an algorithm couldn't find a solution.
cutoff  = Node('cutoff',  path_cost=math.inf) # Indicates iterative deepening search was cut off.
    
    
def expand(problem, node):
    "Expand a node, generating the children nodes."
    s = node.state
    for action in problem.actions(s):
        s1 = problem.result(s, action)
        cost = node.path_cost + problem.action_cost(s, action, s1)
        yield Node(s1, node, action, cost)

# Adjust path_actions to accumulate directional actions
def path_actions(node):
    if node.parent is None:
        return []
    else:
        return path_actions(node.parent) + [node.action]

def path_states(node):
    "The sequence of states to get to this node."
    if node in (cutoff, failure, None): 
        return []
    return path_states(node.parent) + [node.state]

FIFOQueue = deque

LIFOQueue = list


class PriorityQueue:
    """A queue in which the item with minimum f(item) is always popped first."""

    def __init__(self, items=(), key=lambda x: x):
        self.key = key
        self.items = []  # a heap of (score, item) pairs
        for item in items:
            self.add(item)

    def add(self, item):
        """Add item to the queuez."""
        pair = (self.key(item), item)
        heapq.heappush(self.items, pair)

    def pop(self):
        """Pop and return the item with min f(item) value."""
        return heapq.heappop(self.items)[1]

    def top(self):
        return self.items[0][1]

    def __len__(self):
        return len(self.items)

def best_first_search(problem, f):
    "Search nodes with minimum f(node) value first."
    node = Node(problem.initial)
    frontier = PriorityQueue([node], key=f)
    problem.node_counter += 1
    reached = {problem.initial: node}
    while frontier:
        node = frontier.pop()
        if problem.is_goal(node.state):
            return node
        for child in expand(problem, node):
            s = child.state
            if s not in reached or child.path_cost < reached[s].path_cost:
                reached[s] = child
                frontier.add(child)
                problem.node_counter += 1
    return failure


def best_first_tree_search(problem, f):
    "A version of best_first_search without the `reached` table."
    frontier = PriorityQueue([Node(problem.initial)], key=f)
    problem.node_counter += 1
    while frontier:
        node = frontier.pop()
        if problem.is_goal(node.state):
            return node
        for child in expand(problem, node):
            if not is_cycle(child):
                frontier.add(child)
                problem.node_counter += 1
    return failure


def g(n):
    return n.path_cost


def astar_search(problem, h=None):
    """Search nodes with minimum f(n) = g(n) + h(n)."""
    h = h or problem.h
    return best_first_search(problem, f=lambda n: g(n) + h(n))


def astar_tree_search(problem, h=None):
    """Search nodes with minimum f(n) = g(n) + h(n), with no `reached` table."""
    h = h or problem.h
    return best_first_tree_search(problem, f=lambda n: g(n) + h(n))


def weighted_astar_search(problem, h=None, weight=1.4):
    """Search nodes with minimum f(n) = g(n) + weight * h(n)."""
    h = h or problem.h
    return best_first_search(problem, f=lambda n: g(n) + weight * h(n))


def is_cycle(node, k=30):
    "Does this node form a cycle of length k or less?"

    def find_cycle(ancestor, k):
        return (
            ancestor is not None
            and k > 0
            and (ancestor.state == node.state or find_cycle(ancestor.parent, k - 1))
        )

    return find_cycle(node.parent, k)


def breadth_first_search(problem):
    "Search shallowest nodes in the search tree first."
    node = Node(problem.initial)
    if problem.is_goal(problem.initial):
        return node
    frontier = FIFOQueue([node])
    problem.node_counter += 1
    reached = {problem.initial}
    while frontier:
        node = frontier.pop()
        for child in expand(problem, node):
            s = child.state
            if problem.is_goal(s):
                return child
            if s not in reached:
                reached.add(s)
                frontier.appendleft(child)
                problem.node_counter += 1
    return failure


def iterative_deepening_search(problem):
    "Do depth-limited search with increasing depth limits."
    for limit in range(1, sys.maxsize):
        result = depth_limited_search(problem, limit)
        if result != cutoff:
            return result


def depth_limited_search(problem, limit=10):
    "Search deepest nodes in the search tree first."
    frontier = LIFOQueue([Node(problem.initial)])
    problem.node_counter += 1
    result = failure
    while frontier:
        node = frontier.pop()
        if problem.is_goal(node.state):
            return node
        elif len(node) >= limit:
            result = cutoff
        elif not is_cycle(node):
            for child in expand(problem, node):
                frontier.append(child)
                problem.node_counter += 1
    return result



def count_rank_inversion(arr):
    if len(arr) <= 1:
        return arr, 0
    mid = len(arr) // 2  # This line should not be indented under the if statement
    left, inv_left = count_rank_inversion(arr[:mid])
    right, inv_right = count_rank_inversion(arr[mid:])
    merged, inv_split = count_split_inversion(left, right)
    total_inversions = inv_left + inv_right + inv_split
    return merged, total_inversions

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

class EightPuzzle(Problem):
    """The problem of sliding tiles numbered from 1 to 8 on a 3x3 board,
    where one of the squares is a blank, trying to reach a goal configuration.
    A board state is represented as a tuple of length 9, where the element at index i
    represents the tile number at index i, or 0 if for the empty square, e.g. the goal:
        1 2 3
        4 5 6 ==> (1, 2, 3, 4, 5, 6, 7, 8, 0)
        7 8 _
    """

    def __init__(self, initial, goal=(1, 2, 3, 4, 5, 6, 7, 8, 0)):
        super().__init__(initial, goal)  # Call the superclass __init__ if necessary
        self.node_counter = 0
        assert self.check_solvable(initial) % 2 == self.check_solvable(goal) % 2, "Initial and goal states must have the same parity."
        self.initial = initial
        self.goal = goal

    def find_blank_space(self, state):
        """Return the index of the blank space (_ or 0)."""
        return state.index(0)

    def actions(self, state):
        """Return the tiles that can move into the blank space."""
        blank_index = self.find_blank_space(state)
        possible_moves = []
        row, col = divmod(blank_index, 3)  # Get row, col of blank space
        
        if row > 0: possible_moves.append(state[blank_index - 3])  # Tile can move down
        if row < 2: possible_moves.append(state[blank_index + 3])  # Tile can move up
        if col > 0: possible_moves.append(state[blank_index - 1])  # Tile can move right
        if col < 2: possible_moves.append(state[blank_index + 1])  # Tile can move left
        
        return [tile for tile in possible_moves if tile != 0]  # Exclude '0' from possible moves

    """
    "U" in this action sequence means the tile below the blank space moved up.
    "D" means the tile above the blank space moved down.
    "L" means the tile to the right of the blank space moved left.
    "R" means the tile to the left of the blank space moved right.
    """

    def result(self, state, action):
        """Return the state that results from moving a tile into the blank space."""
        blank_index = self.find_blank_space(state)
        tile_index = state.index(action)  # Find the index of the tile to move
        
        # Swap the tile and the blank space
        new_state = list(state)
        new_state[blank_index], new_state[tile_index] = new_state[tile_index], new_state[blank_index]
        
        return tuple(new_state)

    def h1(self, node):
        """The misplaced tiles heuristic."""
        return hamming_distance(node.state, self.goal)

    def h2(self, node):
        """The Manhattan heuristic."""
        X = (0, 1, 2, 0, 1, 2, 0, 1, 2)
        Y = (0, 0, 0, 1, 1, 1, 2, 2, 2)
        return sum(
            abs(X[s] - X[g]) + abs(Y[s] - Y[g])
            for (s, g) in zip(node.state, self.goal)
            if s != 0
        )

    def h3(self, node):
        return self.linear_conflict(node) 

    def check_solvable(self, puzzle):
        # Exclude the empty tile (0) before calculating inversions
        puzzle_without_empty = [num for num in puzzle if num != 0]
        _, total_inversions = count_rank_inversion(puzzle_without_empty)
        return total_inversions

    def linear_conflict(self, node):
        """Calculate the total linear conflict in rows and columns."""
        state = node.state
        total_conflict = 0
        
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                # Check for conflict in rows
                if (state[i] != 0 and state[j] != 0 and
                        (i // 3 == j // 3) and
                        (i // 3 == self.goal.index(state[i]) // 3) and
                        (self.goal.index(state[i]) > self.goal.index(state[j])) == (i < j)):
                    total_conflict += 1

                # Check for conflict in columns
                if (state[i] != 0 and state[j] != 0 and
                        (i % 3 == j % 3) and
                        (i % 3 == self.goal.index(state[i]) % 3) and
                        (self.goal.index(state[i]) > self.goal.index(state[j])) == (i < j)):
                    total_conflict += 1

        # Combine linear conflict with Manhattan distance
        return self.manhattan_distance(node) + 2 * total_conflict

    def manhattan_distance(self, node):
        """Calculate the Manhattan distance from the current state to the goal state."""
        state = node.state
        distance = 0
        for i in range(1, 9):  # Exclude the empty tile
            xi, yi = divmod(state.index(i), 3)
            xg, yg = divmod(self.goal.index(i), 3)
            distance += abs(xi - xg) + abs(yi - yg)
        return distance


# def manhattan_distance(state, goal):
#     distance = 0
#     for i in range(1, 9):  # Exclude the empty tile
#         xi, yi = divmod(state.index(i), 3)
#         xg, yg = divmod(goal.index(i), 3)
#         distance += abs(xi - xg) + abs(yi - yg)
#     return distance



def hamming_distance(A, B):
    "Number of positions where vectors A and B are different."
    return sum(a != b for a, b in zip(A, B))


# def inversions(board):
#     "The number of times a piece is a smaller number than a following piece."
#     return sum((a > b and a != 0 and b != 0) for (a, b) in combinations(board, 2))



def board8(board, fmt=(3 * "{} {} {}\n")):
    "A string representing an 8-puzzle board"
    return fmt.format(*board).replace("0", "_")


class Board(defaultdict):
    empty = "."
    off = "#"

    def __init__(self, board=None, width=8, height=8, to_move=None, **kwds):
        if board is not None:
            self.update(board)
            self.width, self.height = (board.width, board.height)
        else:
            self.width, self.height = (width, height)
        self.to_move = to_move

    def __missing__(self, key):
        x, y = key
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return self.off
        else:
            return self.empty

    def __repr__(self):
        def row(y):
            return " ".join(self[x, y] for x in range(self.width))

        return "\n".join(row(y) for y in range(self.height))

    def __hash__(self):
        return hash(tuple(sorted(self.items()))) + hash(self.to_move)

# Some specific EightPuzzle problems

# e1 = EightPuzzle((1, 4, 2, 0, 7, 5, 3, 6, 8))
# e2 = EightPuzzle((1, 2, 3, 4, 5, 6, 7, 8, 0))
# e3 = EightPuzzle((4, 0, 2, 5, 1, 3, 7, 8, 6))
# e4 = EightPuzzle((7, 2, 4, 5, 0, 6, 8, 3, 1))
# e5 = EightPuzzle((8, 6, 7, 2, 5, 4, 3, 0, 1))


# Read input
def get_puzzle(file_path):
    try:
        with open(file_path, "r") as inFile:
            puzzle = []
            for line in inFile:
                puzzle.extend(line.split())
            puzzle = [int(x.replace("_", "0")) for x in puzzle]
            return tuple(puzzle)
    except FileNotFoundError:
        print("\nFile not found!\n")
        return None


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

# Solve an 8 puzzle problem and print out each state

# for s in path_states(astar_search(e1)):
#     print(board8(s))

class CountCalls:
    """Delegate all attribute gets to the object, and count them in ._counts"""

    def __init__(self, obj):
        self._object = obj
        self._counts = Counter()

    def __getattr__(self, attr):
        "Delegate to the original object, after incrementing a counter."
        self._counts[attr] += 1
        return getattr(self._object, attr)


def report(searchers, problems, verbose=True):
    """Show summary statistics for each searcher (and on each problem unless verbose is false)."""
    for searcher in searchers:
        print(searcher.__name__ + ":")
        total_counts = Counter()
        for p in problems:
            prob = CountCalls(p)
            soln = searcher(prob)
            counts = prob._counts
            counts.update(actions=len(soln), cost=soln.path_cost)
            total_counts += counts
            if verbose:
                report_counts(counts, str(p)[:40])
        report_counts(total_counts, "TOTAL\n")


def report_counts(counts, name):
    """Print one line of the counts report."""
    print(
        "{:9,d} nodes |{:9,d} goal |{:5.0f} cost |{:8,d} actions | {}".format(
            counts["result"], counts["is_goal"], counts["cost"], counts["actions"], name
        )
    )

# def astar_misplaced_tiles(problem):
#     return astar_search(problem, h=problem.h1)


# report(
#     [breadth_first_search, astar_misplaced_tiles, astar_search], [e1, e2, e3, e4, e5]
# )

# report([astar_search, astar_tree_search], [e1, e2, e3, e4])

# def build_table(table, depth, state, problem):
#     if depth > 0 and state not in table:
#         problem.initial = state
#         table[state] = len(astar_search(problem))
#         for a in problem.actions(state):
#             build_table(table, depth - 1, problem.result(state, a), problem)
#     return table


# def invert_table(table):
#     result = defaultdict(list)
#     for key, val in table.items():
#         result[val].append(key)
#     return result


# goal = (1, 2, 3, 4, 5, 6, 7, 8, 0)
# table8 = invert_table(build_table({}, 25, goal, EightPuzzle(goal)))

# def report8(
#     table8,
#     M,
#     Ds=range(2, 25, 2),
#     searchers=(breadth_first_search, astar_misplaced_tiles, astar_search),
# ):
#     "Make a table of average nodes generated and effective branching factor"
#     for d in Ds:
#         line = [d]
#         N = min(M, len(table8[d]))
#         states = random.sample(table8[d], N)
#         for searcher in searchers:
#             nodes = 0
#             for s in states:
#                 problem = CountCalls(EightPuzzle(s))
#                 searcher(problem)
#                 nodes += problem._counts["result"]
#             nodes = int(round(nodes / N))
#             line.append(nodes)
#         line.extend([ebf(d, n) for n in line[1:]])
#         print("{:2} & {:6} & {:5} & {:5} && {:.2f} & {:.2f} & {:.2f}".format(*line))


# def ebf(d, N, possible_bs=[b / 100 for b in range(100, 300)]):
#     "Effective Branching Factor"
#     return min(possible_bs, key=lambda b: abs(N - sum(b**i for i in range(1, d + 1))))


# def edepth_reduction(d, N, b=2.67):


# def random_state():
#     x = list(range(9))
#     random.shuffle(x)
#     return tuple(x)


# meanbf = mean(len(e3.actions(random_state())) for _ in range(10000))
# meanbf

def heuristic_h1(problem):
    return astar_search(problem, h=problem.h1)
def heuristic_h2(problem):
    return astar_search(problem, h=problem.h2)
def heuristic_h3(problem):
    return astar_search(problem, h=problem.h3)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Solve the 8-puzzle problem.')
    parser.add_argument('--fPath', type=str, help='File path of the puzzle.')
    parser.add_argument('--alg', type=str, choices=['BFS', 'IDS', 'h1', 'h2', 'h3'], help='Algorithm.')
    return parser.parse_args()


def run_search_algorithm(search_func, problem, result_queue):
    start_time = time.time()
    try:
        # Initialize the node counter in the problem instance
        problem.node_counter = 0
        solution = search_func(problem)

        # Calculate the total time taken for the search
        total_time = time.time() - start_time

        if solution in [failure, cutoff]:
            # The search failed to find a solution
            result_queue.put({
                "status": "failed",
                "nodes_generated": problem.node_counter,
                "total_time": total_time,
                "path": [],
            })
        else:
            # A solution was found
            path = path_actions(solution)  # Convert the solution path to a sequence of actions
            result_queue.put({
                "status": "solved",
                "nodes_generated": problem.node_counter,
                "total_time": total_time,
                "path": path,
            })
    except Exception as e:
        # In case of any exception, send it back to the main process
        result_queue.put({
            "status": "error",
            "message": str(e),
        })


algorithm_map = {
    'BFS': breadth_first_search,
    'IDS': iterative_deepening_search,
    'h1': heuristic_h1,
    'h2': heuristic_h2,
    'h3': heuristic_h3,
}


def main():
    args = parse_arguments()

    # Load the puzzle from the specified file path
    puzzle = get_puzzle(args.fPath)
    if puzzle is None:
        print("Could not load the puzzle.")
        return

    puzzle_instance = EightPuzzle(puzzle)
    
    # Check if the puzzle is solvable
    if puzzle_instance.check_solvable(puzzle) % 2 != 0:
        print("The inputted puzzle is not solvable:")
        board8(puzzle)
        return

    search_func = algorithm_map.get(args.alg)
    if search_func is None:
        print("Invalid algorithm specified.")
        return

    result_queue = Queue()
    start_time = time.time()

    # Create and start the search process
    search_process = Process(target=run_search_algorithm, args=(search_func, puzzle_instance, result_queue))
    search_process.start()
    search_process.join(timeout=900)  # 15 minutes

    total_time = time.time() - start_time
    seconds = int(total_time)
    microseconds = int((total_time - seconds) * 1_000_000)

    if search_process.is_alive():
        # If process is still alive after the timeout
        search_process.terminate()
        search_process.join()
        print(f"Algorithm timed out after 15 minutes. Total time taken: >15 min")
        print("Total nodes generated: ", "Data unavailable — process was terminated")
        print("Path length: Timed out.")
        print("Path: Timed out.")
    else:
        # Process completed within the time limit
        try:
            result = result_queue.get_nowait()
            if isinstance(result, Exception):
                raise result
            elif result == failure or result == cutoff:
                print("No solution found or search was cut off.")
            else:
                print(f"Total nodes generated: {result['nodes_generated']}")
                print(f"Total time taken: {seconds} sec {microseconds} microSec.")
                print(f"Path length: {len(result['path'])}")
                print(f"Path: {''.join(result['path'])}")
        except queue.Empty:
            print("No result was returned by the search algorithm.")

if __name__ == '__main__':
    main()
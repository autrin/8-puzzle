import argparse
import time
import matplotlib.pyplot as plt
import random
import heapq
import math
import sys
from collections import defaultdict, deque, Counter
from itertools import combinations
from statistics import mean 
from functools import lru_cache
from multiprocessing import Process, Queue

class Problem(object):
    """The abstract class for a formal problem. A new domain subclasses this,
    overriding `actions` and `results`, and perhaps other methods.
    The default heuristic is 0 and the default action cost is 1 for all states.
    When yiou create an instance of a subclass, specify `initial`, and `goal` states 
    (or give an `is_goal` method) and perhaps other keyword args for the subclass."""

    def __init__(self, initial=None, goal=None, **kwds): 
        self.__dict__.update(initial=initial, goal=goal, **kwds) 
        
    def actions(self, state):        raise NotImplementedError
    def result(self, state, action): raise NotImplementedError
    def is_goal(self, state):        return state == self.goal
    def action_cost(self, s, a, s1): return 1 # Assuming the cost is 1 for every action/move
    def h(self, node):               return 0
    
    def __str__(self):
        return '{}({!r}, {!r})'.format(
            type(self).__name__, self.initial, self.goal)
    
class Node:
    "A Node in a search tree."
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.__dict__.update(state=state, parent=parent, action=action, path_cost=path_cost)

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
        
# def path_actions(node):
#     """Return a string of actions to get to this node."""
#     if node.parent is None:
#         return ""
#     actions = path_actions(node.parent) + [node.action]
#     # Map the internal action representation to the project's specified format
#     action_map = {
#         (0, -1): 'U',
#         (0, 1): 'D',
#         (-1, 0): 'L',
#         (1, 0): 'R',
#     }
#     return ''.join(action_map[action] for action in actions if action in action_map)

def path_actions(node):
    """Generate a sequence of number tile movements that led to this node."""
    if node.parent is None:
        return ""
    else:
        # Assuming node.action now represents the tile that moved
        return path_actions(node.parent) + str(node.action)


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
    return failure


def best_first_tree_search(problem, f):
    "A version of best_first_search without the `reached` table."
    frontier = PriorityQueue([Node(problem.initial)], key=f)
    while frontier:
        node = frontier.pop()
        if problem.is_goal(node.state):
            return node
        for child in expand(problem, node):
            if not is_cycle(child):
                frontier.add(child)
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
        # if not self.check_solvable(initial):
        #     print("The puzzle is unsolvable.")
        #     sys.exit()
        assert self.check_solvable(initial) % 2 == self.check_solvable(self.goal) % 2  # Parity check. Initial and goal states must have the same parity.
        # self.initial, self.goal = initial, goal
        super().__init__(initial, goal)

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

    # def result(self, state, action):
    #     """Swap the blank with the square numbered `action`."""
    #     s = list(state)
    #     blank = state.index(0)
    #     s[action], s[blank] = s[blank], s[action]
    #     return tuple(s)

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
        return total_inversions % 2 == 0

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

e1 = EightPuzzle((1, 4, 2, 0, 7, 5, 3, 6, 8))
e2 = EightPuzzle((1, 2, 3, 4, 5, 6, 7, 8, 0))
e3 = EightPuzzle((4, 0, 2, 5, 1, 3, 7, 8, 6))
e4 = EightPuzzle((7, 2, 4, 5, 0, 6, 8, 3, 1))
e5 = EightPuzzle((8, 6, 7, 2, 5, 4, 3, 0, 1))


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

for s in path_states(astar_search(e1)):
    print(board8(s))

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

def astar_misplaced_tiles(problem):
    return astar_search(problem, h=problem.h1)


report(
    [breadth_first_search, astar_misplaced_tiles, astar_search], [e1, e2, e3, e4, e5]
)

report([astar_search, astar_tree_search], [e1, e2, e3, e4])

def build_table(table, depth, state, problem):
    if depth > 0 and state not in table:
        problem.initial = state
        table[state] = len(astar_search(problem))
        for a in problem.actions(state):
            build_table(table, depth - 1, problem.result(state, a), problem)
    return table


def invert_table(table):
    result = defaultdict(list)
    for key, val in table.items():
        result[val].append(key)
    return result


goal = (1, 2, 3, 4, 5, 6, 7, 8, 0)
table8 = invert_table(build_table({}, 25, goal, EightPuzzle(goal)))

def report8(
    table8,
    M,
    Ds=range(2, 25, 2),
    searchers=(breadth_first_search, astar_misplaced_tiles, astar_search),
):
    "Make a table of average nodes generated and effective branching factor"
    for d in Ds:
        line = [d]
        N = min(M, len(table8[d]))
        states = random.sample(table8[d], N)
        for searcher in searchers:
            nodes = 0
            for s in states:
                problem = CountCalls(EightPuzzle(s))
                searcher(problem)
                nodes += problem._counts["result"]
            nodes = int(round(nodes / N))
            line.append(nodes)
        line.extend([ebf(d, n) for n in line[1:]])
        print("{:2} & {:6} & {:5} & {:5} && {:.2f} & {:.2f} & {:.2f}".format(*line))


def ebf(d, N, possible_bs=[b / 100 for b in range(100, 300)]):
    "Effective Branching Factor"
    return min(possible_bs, key=lambda b: abs(N - sum(b**i for i in range(1, d + 1))))


# def edepth_reduction(d, N, b=2.67):


def random_state():
    x = list(range(9))
    random.shuffle(x)
    return tuple(x)


meanbf = mean(len(e3.actions(random_state())) for _ in range(10000))
# meanbf


def parse_arguments():
    parser = argparse.ArgumentParser(description='Solve the 8-puzzle problem with various algorithms.')
    parser.add_argument('--fPath', type=str, help='File path of the puzzle to solve.')
    parser.add_argument('--alg', type=str, choices=['BFS', 'IDS', 'h1', 'h2', 'h3'], help='Algorithm to use for solving.')
    return parser.parse_args()

def run_search_algorithm(search_func, problem, result_queue):
    try:
        result = search_func(problem)
        result_queue.put(result)  # Store the result in the queue
    except Exception as e:
        result_queue.put(e)  # Store the exception if something went wrong

def main():
    args = parse_arguments()

    # Load the puzzle from the specified file path
    puzzle = get_puzzle(args.fPath)
    if puzzle is None:
        print("Could not load the puzzle.")
        return  # Exit if the puzzle could not be loaded

    puzzle_instance = EightPuzzle(puzzle)
    
    # Check if the puzzle is solvable
    if not puzzle_instance.check_solvable(puzzle):
        print("The puzzle is not solvable.")
        return

    # Mapping from algorithm name to function
    algorithm_map = {
        'BFS': breadth_first_search,
        'IDS': iterative_deepening_search,
        'h1': lambda p: astar_search(p, h=p.h1),
        'h2': lambda p: astar_search(p, h=p.h2),
        'h3': lambda p: astar_search(p, h=p.h3),
    }

    search_func = algorithm_map.get(args.alg)
    if search_func is None:
        print("Invalid algorithm specified.")
        return

    # Create a queue to store the result of the search algorithm
    result_queue = Queue()

    # Start measuring time
    start_time = time.time()

    # Create and start the search process
    search_process = Process(target=run_search_algorithm, args=(search_func, puzzle_instance, result_queue))
    search_process.start()

    # Wait for the process to complete or timeout
    search_process.join(timeout=900)  # 15 minutes expressed in seconds

    # Stop measuring time
    end_time = time.time()
    total_time = end_time - start_time

    # Check if the process is still alive (which means it timed out)
    if search_process.is_alive():
        search_process.terminate()
        search_process.join()
        print("Algorithm timed out after 15 minutes.")
    else:
        # Process completed within the time limit, get the result
        result = result_queue.get()
        if isinstance(result, Exception):
            raise result
        elif result == failure:
            print("No solution found.")
        elif result == cutoff:
            print("Search was cut off.")
        else:
            # Output the results
            print(f"Total nodes generated: {len(path_actions(result))}")
            print(f"Total time taken: {total_time:.2f} seconds")
            path = path_actions(result)  # Get the actions that led to the solution
            print(f"Path: {''.join(path)}")

if __name__ == '__main__':
    main()
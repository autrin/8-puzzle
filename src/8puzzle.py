# This program was written with the help of algorithms from AIMA 4th edition.

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
from os import walk
from os.path import join, basename, exists


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
        # Directly check the value of node.action without wrapping it in a list
        if node.action == 'U':
            action = 'D'  # If the blank moves up, the tile moves down
        elif node.action == 'D':
            action = 'U'  # If the blank moves down, the tile moves up
        elif node.action == 'L':
            action = 'R'  # If the blank moves left, the tile moves right
        elif node.action == 'R':
            action = 'L'  # If the blank moves right, the tile moves left
        else:
            action = node.action  # This else clause might be unnecessary, just for safety
        
        # Recursively accumulate actions, add the current action as part of the list
        return path_actions(node.parent) + [action]


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
        assert self.check_solvable(initial) % 2 == self.check_solvable(goal) % 2, "The inputted puzzle is not solvable."
        self.initial = initial
        self.goal = goal

    def find_blank_space(self, state):
        """Return the index of the blank space (_ or 0)."""
        return state.index(0)

    def actions(self, state):
        """Return the actions that can be executed in the given state."""
        possible_actions = []
        blank_index = state.index(0)
        row, col = divmod(blank_index, 3)

        if row > 0: possible_actions.append('U')  # Blank can move up
        if row < 2: possible_actions.append('D')  # Blank can move down
        if col > 0: possible_actions.append('L')  # Blank can move left
        if col < 2: possible_actions.append('R')  # Blank can move right

        return possible_actions

    def result(self, state, action):
        """Return the resulting state from taking action in state."""
        blank_index = state.index(0)
        new_state = list(state)
        """
        "U" in this action sequence means the tile below the blank space moved up.
        "D" means the tile above the blank space moved down.
        "L" means the tile to the right of the blank space moved left.
        "R" means the tile to the left of the blank space moved right.
        """
        if action == 'U': swap_with = blank_index - 3
        elif action == 'D': swap_with = blank_index + 3
        elif action == 'L': swap_with = blank_index - 1
        elif action == 'R': swap_with = blank_index + 1

        new_state[blank_index], new_state[swap_with] = new_state[swap_with], new_state[blank_index]
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

def hamming_distance(A, B):
    "Number of positions where vectors A and B are different."
    return sum(a != b for a, b in zip(A, B))


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


def heuristic_h1(problem):
    return astar_search(problem, h=problem.h1)
def heuristic_h2(problem):
    return astar_search(problem, h=problem.h2)
def heuristic_h3(problem):
    return astar_search(problem, h=problem.h3)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Solve the 8-puzzle problem.')
    parser.add_argument('--fPath', type=str, help='File path of the puzzle.')
    parser.add_argument('--alg', type=str, choices=['BFS', 'IDS', 'h1', 'h2', 'h3', 'all'], help='Algorithm.')
    parser.add_argument('--part', type=str, choices=['2', '3'], help='Specify which part of the project to run.')
    return parser.parse_args()


def run_search_algorithm(search_func, problem, result_queue, solved_count):
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

    #? I can use the result queue for part 3

algorithm_map = {
    'BFS': (breadth_first_search, "Breadth First Search"),
    'IDS': (iterative_deepening_search, "Iterative Deepening Search"),
    'h1': (heuristic_h1, "Heuristic H1"),
    'h2': (heuristic_h2, "Heuristic H2"),
    'h3': (heuristic_h3, "Heuristic H3"),
}

def main(search_func, algorithm_name, total_time3, total_nodes3, solved_count, timeouts, file_path):
    if args.part == '3' and args.alg == 'all':
        # Ensure the 'Experiment_results' directory exists
        results_dir = 'Experiment_results3'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)  # Creates the directory if it does not exist
        output_file_name = 'part3_results.txt'
        global output_file_path
        output_file_path = os.path.join(results_dir, output_file_name)  # Construct the full path
        # args.fPath = file_path
        search_func = search_func3
    elif args.part == '2' and args.alg in algorithm_map:
        results_dir = 'Test_results2'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        # file_name = os.path.basename(args.fPath)  # Extract the filename from the path
        output_file_name = 'part2_results.txt'
        output_file_path = os.path.join(results_dir, output_file_name)
        
    # Load the puzzle from the specified file path
    puzzle = get_puzzle(file_path)
    if puzzle is None:
        with open(output_file_path, 'a') as file:
            file.write("Could not load the puzzle.\n")
        print("Could not load the puzzle.")
        return
    try:
        puzzle_instance = EightPuzzle(puzzle) # It will check inversion rank % 2 == 0
    except AssertionError as error:
        with open(output_file_path, 'a') as file:
            file.write(f"\nUnsolvable puzzle in file {file_path}: {error}")
        print(f"\nUnsolvable puzzle at {file_path}.")
        return
    if args.part == 2:
        search_func, algorithm_name = algorithm_map.get(args.alg, (None, "Unknown Algorithm"))
    if search_func is None:
        with open(output_file_path, 'a') as file:
            file.write(f"\nInvalid algorithm specified at {file_path}.\n")
        print("\nInvalid algorithm specified.")
        return

    result_queue = Queue()
    start_time = time.time()

    # Create and start the search process
    search_process = Process(target=run_search_algorithm, args=(search_func, puzzle_instance, result_queue, solved_count))
    search_process.start()
    search_process.join(timeout=900)  # 15 minutes

    total_time = time.time() - start_time
    seconds = int(total_time)  # Whole seconds part
    fractional_seconds = total_time - seconds  # Fractional part of the seconds
    microseconds = int(fractional_seconds * 1_000_000)  # Convert fractional seconds to microseconds

    # Write results to file
    with open(output_file_path, 'a') as file:
        if search_process.is_alive():
            # If process is still alive after the timeout
            search_process.terminate()
            timeouts += 1  # Track timeouts for additional insights
            search_process.join()
            file.write(f"Algorithm timed out after 15 minutes. Total time taken: >15 min\n")
            file.write("Total nodes generated: Data unavailable — process was terminated\n")
            file.write("Path length: Timed out.\n")
            file.write("Path: Timed out.\n")
            file.write(f"Algorithm was: {algorithm_name}\n")
            file.write(f"Puzzle was at {file_path}\n")
        else:
            # Process completed within the time limit
            try:
                result = result_queue.get_nowait()
                if isinstance(result, Exception):
                    raise result
                elif result == failure or result == cutoff:
                    file.write("No solution found or search was cut off.\n")
                else:
                    file.write(f"\nTotal nodes generated: {result['nodes_generated']}\n")
                    file.write(f"Total time taken: {seconds} sec {microseconds} microSec.\n")
                    file.write(f"Path length: {len(result['path'])}\n")
                    file.write(f"Path: {''.join(result['path'])}\n")
                    file.write(f"Algorithm was: {algorithm_name}\n")
                    file.write(f"Puzzle was at {file_path}")
                    if args.part == '3' and args.alg == 'all':
                        total_time3 += total_time
                        total_nodes3 += result['nodes_generated']
                        solved_count += 1
            except queue.Empty:
                file.write("No result was returned by the search algorithm.\n")
                file.write(f"Puzzle was at {file_path}")
    print(f"Results written to {output_file_path}")
    return (total_nodes3, total_time3, solved_count)



if __name__ == '__main__':
    args = parse_arguments()

    total_time3, total_nodes3, solved_count, timeouts = 0,0,0,0

    if args.part == '2':
        if args.alg in algorithm_map:
            search_func, algorithm_name = algorithm_map[args.alg]
            _, _, _ = main(search_func, algorithm_name, total_time3, total_nodes3, solved_count, timeouts, args.fPath)
        else:
            print("Please select an algorithm among 'BFS', 'IDS', 'h1', 'h2', 'h3'")
    elif args.part == '3' and args.alg == 'all':
        for subdir in ['L8', 'L15', 'L24']:
            folder_path = join(args.fPath, subdir)
            for alg_name, (search_func3, alg_desc) in algorithm_map.items():
                result3 = {}
                for subdir1, dirs, files in walk(folder_path):
                    for filename in files:
                        if filename.endswith(".txt"):
                            file_path = join(subdir1, filename)
                            total_nodes3, total_time3, solved_count = main(search_func3, alg_name, total_time3, total_nodes3, solved_count, timeouts, file_path)
                            
                    # Store averaged results for this algorithm
                    if solved_count > 0:
                        avg_time = (total_time3 * 1_000_000) / solved_count
                        avg_nodes = total_nodes3 / solved_count
                    else:
                        avg_time, avg_nodes = 0, 0
                    result3[alg_desc] = (avg_time, avg_nodes, solved_count, timeouts)
                    for alg_desc, metrics in result3.items():
                        with open(output_file_path, 'a') as file:
                            file.write(f"\n******************\nDepth: {subdir}, {alg_desc} - Average Time: {metrics[0]}microsconds, Average Nodes: {metrics[1]}, Puzzles Solved: {metrics[2]}, Timeouts: {metrics[3]}\n******************\n")
                    total_time3, total_nodes3, solved_count, timeouts = 0,0,0,0

    else:
        print("Please specify a valid part (2 or 3) and algorithm.")
# 8-Puzzle Solver 

An AI-based solver for the classic 8-puzzle problem

## Algorithms

- **BFS**: Explores nodes level by level for guaranteed shortest path.
- **IDS**: Combines DFS space efficiency with BFS optimality.
- **A***: Uses heuristics for efficient optimal solutions:
  - **Manhattan Distance**: Sum of distances of tiles from goal positions.
  - **Linear Conflict**: Manhattan Distance plus additional penalties for tiles in conflict.

## Performance

- Achieves up to 60x speedup by leveraging Python's multiprocessing.
- Capable of solving hundreds of puzzles in 10 minutes.

## Overview

This project provides a flexible solver for the 8-puzzle, implementing Breadth-First Search (BFS), Iterative Deepening Search (IDS), and A* Search with both Manhattan Distance and Linear Conflict heuristics.
It is designed for efficiency and scalability, solving hundreds of puzzles in just 10 minutes thanks to advanced optimizations.

## Key Features

- AI Search Algorithms: BFS, IDS, and A* with Manhattan Distance & Linear Conflict heuristics.
- Performance Optimized: Achieves a 60x speedup through multiprocessing; processes large puzzle batches rapidly.
- Flexible Usage: Run on a single test file or entire directories for batch solving.
- Pythonic Design: Built using Python and NumPy.

## Tech Stack

- Python
- NumPy
- AI Search Algorithms (BFS, IDS, A* with advanced heuristics)
- Multiprocessing

## Getting Started

### 1. Clone the Repository

```sh
git clone https://github.com/autrin/8-puzzle.git
cd 8-puzzle
```

### 2. Prepare Test Files

Place test files in the `test/Part2` or `test/Part3` directories, or adjust paths as needed.

## Usage

### Solve a Single Test File

```sh
python3 8puzzle.py --fPath <file.txt> --alg <Algorithm> --part 2
```

- `<file.txt>`: Path to the test file (e.g., test/Part2/S1.txt)
- `<Algorithm>`: Algorithm to use (bfs, ids, h1, h2, etc.)
- `--part 2`: Required for single test mode

#### Example

```sh
python3 8puzzle.py --fPath test/Part2/S1.txt --alg h1 --part 2
```

### Solve All Test Files in a Directory

```sh
python3 8puzzle.py --fPath <Part3 path> --alg all --part 3
```

- `<Part3 path>`: Directory containing .txt files (e.g., test/Part3)
- `--alg all`: Runs all algorithms
- `--part 3`: Required for batch mode

#### Example

```sh
python3 8puzzle.py --fPath test/Part3 --alg all --part 3
```

## Notes

- Always include `--part 2` or `--part 3` as appropriate.
- Adjust file paths if your directory structure differs.

## License

MIT License.
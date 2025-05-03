# Testing Guide for Python Code Obfuscator

This guide will walk you through testing the code obfuscator with one of the repositories mentioned in the requirements.

## Prerequisites

- Python 3.6 or higher
- Git
- The code obfuscator installed (`pip install -e .` from the repository root)

## Step 1: Clone a Test Repository

Choose one of the repositories mentioned in the requirements:

```bash
# Clone a sample repository
git clone https://github.com/dhhruv/Sudoku-Solver.git
cd Sudoku-Solver

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Test the Original Repository

Make sure the original repository works:

```bash
# Run tests if they exist, or a sample program from the repository
python solve.py
```

## Step 3: Obfuscate the Repository

Use the code obfuscator to create an obfuscated version of the repository:

```bash
# Create a directory for the obfuscated code
mkdir -p ../Sudoku-Solver-Obfuscated

# Run the obfuscator
obfuscate Sudoku-Solver/ ../Sudoku-Solver-Obfuscated/ --keep-structure
```

## Step 4: Test the Obfuscated Repository

Make sure the obfuscated repository works:

```bash
# Navigate to the obfuscated repository
cd ../Sudoku-Solver-Obfuscated

# Install dependencies
pip install -r requirements.txt

# Run tests if they exist, or a sample program from the repository
python solve.py
```

## Step 5: Compare the Original and Obfuscated Code

Compare the original and obfuscated code to see the differences:

```bash
# Compare a specific file
diff -u ../Sudoku-Solver/solve.py solve.py
```

You should see that all custom identifiers have been replaced with random strings, while imported modules, keywords, and comments remain unchanged.

## Testing with Other Repositories

You can follow the same process for any of the repositories mentioned in the requirements:

- [dhhruv/Sudoku-Solver](https://github.com/dhhruv/Sudoku-Solver)
- [francoischalifour/todo-cli](https://github.com/francoischalifour/todo-cli)
- [Kalebu/Plagiarism-checker-Python](https://github.com/Kalebu/Plagiarism-checker-Python)
- [kishanrajput23/Jarvis-Desktop-Voice-Assistant](https://github.com/kishanrajput23/Jarvis-Desktop-Voice-Assistant)
- [tfeldmann/organize](https://github.com/tfeldmann/organize)
- [techwithtim/3-Mini-Python-Projects](https://github.com/techwithtim/3-Mini-Python-Projects/blob/main/project1.py)
- [alfredodeza/basic-python-cli](https://github.com/alfredodeza/basic-python-cli)
- [DebRC/Algorithm-Visualizer](https://github.com/DebRC/Algorithm-Visualizer)

## Troubleshooting

If the obfuscated code doesn't work:

1. Check for errors in the console output
2. Ensure all dependencies are installed
3. Look for issues with imports or file paths
4. Verify that the obfuscator is not replacing imported module names
5. Check for special Python constructs that might not be handled correctly by the obfuscator

If you find any issues, please report them on the issue tracker. 
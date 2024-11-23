# WAT.ai x Hamming.ai Project

### Conda Environment

- Make sure you have a conda environment installed on local machine before proceeding.
- Run the following commands to install necessary packages.
```
!conda create --name watai_hammingai_project python=3.10
!conda activate watai_hammingai_project
!conda install datasets
!pip install tiktoken litellm
!cp .env.example .env
!pip install -e .
```
- Make sure to replace the environment variable values in `.env` file with valid API keys.

- After the installation is finished, run the following commands to get started on a terminal.
```
# Initializes conda environment
!conda activate watai_hammingai_project
# Initializes environment variables
!source .env
```

### Pull Request

- To push code into the GitHub repo, please create a new git branch and push the new branch to Github, then create a pull request.
- The pull request requires an approval from 1 additional member before being merged to the `main` branch.
```
!git checkout -b <branch-name>
# add, commit
!git push -u origin <branch-name>
```

### Demo (MBPP)

- Codebase for running code-generation on MBPP benchmark and evaluating the results against the unit tests.
- Create a directory called `eval_results` under `/demo` before running the following commands.
- Navigate to `/demo` directory: `!cd demo`.
- Run the following command to run the `gpt-4o-mini` on MBPP train/test datasets (you can change the model in the source code).
```
!python3 codegen.py \
    --start <start_index> \ # Row index to start from (inclusive)
    --end <end_index> \ # Row index to end at (exclusive)
    --section <optional-section> \ # Optional section of the MBPP dataset (test or train, defaults to test)
    --version <optional-version> # Optional version prefix for the output file
```
- Run the folllowing command to evaluate the results stored under `/demo/eval_results` directory.
```
!python3 codeval.py \
    --src_file <src_file> \ # Source JSONL file containing models to evaluate
```

### o1 Research

Contains research work for reverse-engineering internal-reasoning models such as the o1 models.

> Working with the codebase

- Run `python3 model.py` to run the baseline reverse-engineered o1 model.
- Create a directory called `eval_results` under `/o1_research` before running the following commands.
- Navigate to `/o1_research` directory: `!cd o1_research`.
- Run the following command to run the baseline model on MBPP train/test datasets.
```
!python3 codegen.py \
    --start <start_index> \ # Row index to start from (inclusive)
    --end <end_index> \ # Row index to end at (exclusive)
    --section <optional-section> \ # Optional section of the MBPP dataset (test or train, defaults to test)
    --version <optional-version> # Optional version prefix for the output file
```
- Run the folllowing command to evaluate the results stored under `/o1_research/eval_results` directory.
```
!python3 codeval.py \
    --src_file <src_file> \ # Source JSONL file containing models to evaluate
    --section <section> # Section of the MBPP dataset (test or train)
```

### o1 Research - Custom Reward Model

- `accuracy_test.py`: script for evaluating model accuracy across a defined set of test cases
- `new_model.py`: new model.py with the reward model implementation
- `simple_testing.py`: lightweight script for running basic tests on the model to verify outputs quickly
- `train_reward_model.py`: script for training the reward model

### Test Results

> MBPP Leaderboard

- MapCoder (o1-mini): `93.20%`
- Monte Carlo Search Tree (o1-mini): `82.30%`
- O1BaselineModel (Hybrid: gpt-4o, claude-3.5-sonnet, gemini-1.5-pro, command-r-plus): `62.53%`
- gpt-4o-mini: `61.00%`
- O1BaselineModel (gpt-4o-mini): `59.80%`

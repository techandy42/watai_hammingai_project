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

- Demo code for generating code for MBPP benchmark and testing the code against the provided unit tests.
- Run `python3 codegen.py` to generate code for the MBPP benchmark. (stores output in `mbpp_hammingai.csv`)
- Run `python3 codeval.py` to validate the code against the unit tests. (stores output in `mbpp_hammingai_validated.csv`)

### o1 Research

- Run `python3 model.py` to run the baseline reverse-engineered o1 model.

### Test Results

> MBPP

- GPT-4o Mini (Basic Structure Guide): `61.00%`
- O1Baseline (GPT-4o Mini) (Basic Structure Guide): `59.80%`

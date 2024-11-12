import argparse
import json
import ast
import pandas as pd
from typing import List
from datasets import load_dataset
from o1_research.model import O1BaselineModel, initialize_models_from_jsonl
from o1_research.helpers import MBPPRequestId
from demo.codegen import add_comma_to_newline, get_split_assert_statements
from demo.codeval import get_code_template, run_code

class CodeValMBPP:
    def __init__(self, models: List[O1BaselineModel], section: str):
        self.models = models
        self.successful_models = []
        self.failed_models = []
        self.section = section

    # Get a list of models that passes the unit tests
    def evaluate(self):
        dataset = load_dataset('google-research-datasets/mbpp')
        test_dataset = dataset[self.section]
        df_all_test = test_dataset.to_pandas()

        df_test = []

        for model in self.models:
            task_id = MBPPRequestId.extract_request_id(model.request_id)
            row = df_all_test[df_all_test['task_id'] == task_id].iloc[0]
            pred_code = model.thought_chain.get_final_answer()
            test_list = row['test_list']
            challenge_test_list = row['challenge_test_list']
            df_test.append({
                'request_id': model.request_id,
                'pred_code': pred_code,
                'test_list': test_list,
                'challenge_test_list': challenge_test_list
            })

        df_test = pd.DataFrame(df_test)
        df_test["test_list"] = df_test.apply(add_comma_to_newline, axis=1, column_name="test_list")
        df_test["challenge_test_list"] = df_test.apply(add_comma_to_newline, axis=1, column_name="challenge_test_list")
        df_test["test_list"] = df_test['test_list'].apply(ast.literal_eval)
        df_test["challenge_test_list"] = df_test['challenge_test_list'].apply(ast.literal_eval)
        df_test["test_list"] = df_test.apply(get_split_assert_statements, axis=1, column_name="test_list")
        df_test["challenge_test_list"] = df_test.apply(get_split_assert_statements, axis=1, column_name="challenge_test_list")
        df_test["code_template"] = df_test.apply(get_code_template, axis=1)
        print("Running unit tests...")
        df_test[['stdout_lines', 'status_code']] = df_test.progress_apply(run_code, axis=1, result_type="expand")
        df_successful_test = df_test[df_test['status_code'] == 1]
        successful_request_ids = set(df_successful_test['request_id'])

        successful_models = []
        failed_models = []
        for model in self.models:
            if model.request_id in successful_request_ids:
                successful_models.append(model)
            else:
                failed_models.append(model)

        self.successful_models = successful_models
        self.failed_models = failed_models

    # Save the list of successful models to a jsonl file
    def save_successful_models(self, file_path: str):
        results = []
        for model in self.successful_models:
            results.append(model.save_result())

        with open(file_path, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')

    def save_failed_models(self, file_path: str):
        results = []
        for model in self.failed_models:
            results.append(model.save_result())

        with open(file_path, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')

    def eval_accuracy(self) -> float:
        return len(self.successful_models) / len(self.models)

# Run codeval
def main():
    parser = argparse.ArgumentParser(description='Evaluate MBPP code generation results')
    parser.add_argument('--src_file', type=str, required=True, help='Source JSONL file containing models to evaluate')
    parser.add_argument('--section', type=str, required=True, help='Section of the MBPP dataset (test or train)')
    
    args = parser.parse_args()

    success_tgt_file = args.src_file.replace(".jsonl", "_successful.jsonl")
    failed_tgt_file = args.src_file.replace(".jsonl", "_failed.jsonl")
    
    models = initialize_models_from_jsonl(f"./eval_results/{args.src_file}")
    code_eval = CodeValMBPP(models=models, section=args.section)
    code_eval.evaluate()
    code_eval.save_successful_models(f"./eval_results/{success_tgt_file}")
    code_eval.save_failed_models(f"./eval_results/{failed_tgt_file}")
    eval_accuracy = code_eval.eval_accuracy()
    print(f"Accuracy: {eval_accuracy*100:.2f}%")

if __name__ == "__main__":
    main()

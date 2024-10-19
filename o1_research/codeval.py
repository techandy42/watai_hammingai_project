import ast
import pandas as pd
from typing import List
from datasets import load_dataset
from o1_research.model import O1BaselineModel, initialize_models_from_jsonl
from demo.codeval import get_code_template, run_code

class CodeEvalMBPP:
    def __init__(self, models: List[O1BaselineModel]):
        self.models = models

    # Return a list of models that passes the unit tests
    def evaluate(self) -> List[O1BaselineModel]:
        dataset = load_dataset('google-research-datasets/mbpp')
        test_dataset = dataset["test"]
        df_all_test = test_dataset.to_pandas()

        df_test = []

        for model in self.models:
            task_id = int(model.request_id.split('_')[1])
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
        df_test["test_list"] = df_test['test_list'].apply(ast.literal_eval)
        df_test["challenge_test_list"] = df_test['challenge_test_list'].apply(ast.literal_eval)
        df_test["code_template"] = df_test.apply(get_code_template, axis=1)
        print("Running unit tests...")
        df_test[['stdout_lines', 'status_code']] = df_test.progress_apply(run_code, axis=1, result_type="expand")
        print(df_test.shape[0])
        print(df_test.iloc[0])

if __name__ == "__main__":
    from example.example import example
    # models = initialize_models_from_jsonl("mbpp_results.jsonl")
    # code_eval = CodeEvalMBPP(models)
    # code_eval.evaluate()

from datasets import load_dataset
import tiktoken

def calculate_token_count(row, column_name, encoding):
    return len(encoding.encode(row[column_name]))

def calc_cost(read_cost: float, write_cost: float, additional_input: int = 0, 
              additional_output: int = 0, additional_overall_input: int = 0,
              additional_overall_output: int = 0) -> float:
    # Load dataset
    dataset = load_dataset('google-research-datasets/mbpp')
    test_dataset = dataset["test"]
    df_test = test_dataset.to_pandas()

    # Calculate token counts using text and code columns directly
    encoding = tiktoken.get_encoding("cl100k_base")
    df_test["input_token_count"] = df_test.apply(
        calculate_token_count, axis=1, column_name="text", encoding=encoding
    ) + additional_input
    df_test["output_token_count"] = df_test.apply(
        calculate_token_count, axis=1, column_name="code", encoding=encoding
    ) + additional_output

    total_read_token_count = df_test["input_token_count"].sum() + additional_overall_input
    total_write_token_count = df_test["output_token_count"].sum() + additional_overall_output

    estimated_cost = (total_read_token_count / 1_000_000) * read_cost + (total_write_token_count / 1_000_000) * write_cost
    return total_read_token_count, total_write_token_count, estimated_cost

if __name__ == "__main__":
    MODEL_NAME = "o1-mini-2024-09-12"
    READ_COST = 3.00
    WRITE_COST = 12.00
    read_token_count, write_token_count, estimated_cost = calc_cost(
        read_cost=READ_COST,
        write_cost=WRITE_COST,
        additional_input=200,
        additional_output=100,
        additional_overall_input=0,
        additional_overall_output=0
    )
    print(f"Model: {MODEL_NAME}")
    print(f"Read: {read_token_count} tokens / Write: {write_token_count} tokens")
    print(f"Cost: ${estimated_cost:.2f} USD")

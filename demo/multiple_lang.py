# Example code to evaluate multiple programming languages
import requests
import time

def run_and_check_results(source_code, expected_output, language_id):
    url = "https://judge0-ce.p.rapidapi.com/submissions"
    headers = {
        'Content-Type': 'application/json',
        'x-rapidapi-host': 'judge0-ce.p.rapidapi.com',
        'x-rapidapi-key': ''  # Replace with RapidAPI key
    }
    
    payload = {
        "source_code": source_code,
        "language_id": language_id,  
        "stdin": '',
        "expected_output": ''
    }
    
    try:
        # Submitting the source code
        response = requests.post(url, json=payload, headers=headers, params={"base64_encoded": "false", "fields": "*"})
        response.raise_for_status()
        submission_token = response.json()['token']
        
        print('Submission successful, retrieving results...')
        
        # Polling for the result
        result_url = f"{url}/{submission_token}"
        while True:
            result_response = requests.get(result_url, headers=headers, params={"base64_encoded": "false", "fields": "*"})
            result_response.raise_for_status()
            result_data = result_response.json()
            
            if result_data['status']['description'] not in ['In Queue', 'Processing']:
                stdout = result_data.get('stdout', '').strip()  

                if stdout == expected_output:
                    return "Test passed"
                else:
                    print("Test failed")
                    print(f"Expected: '{expected_output}', Got: '{stdout}'")
                    return "Test failed"
                
            time.sleep(1)  # Delay to avoid rate limit or high frequency polling
            
    except requests.exceptions.RequestException as e:
        print('Failed to process the submission:', e)
        return "Test failed"

# Example usage:
language_id = 63  # Addlanguage_id 
source_code = "function add(a, b) { return a + b; }\nconsole.log(add(5, 3));"
expected_output = "9"  

print(run_and_check_results(source_code, expected_output, language_id))


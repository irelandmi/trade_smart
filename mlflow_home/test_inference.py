# test_inference.py
import requests

def main():
    # The URL of your locally served MLflow model endpoint
    url = "http://127.0.0.1:8080/invocations"

    # The 'dataframe_split' format expects:
    # {
    #   "dataframe_split": {
    #       "columns": [...],
    #       "data": [...]
    #   }
    # }
    # where "data" is a list of lists (one sub-list per row).
    sample_input = {
        "dataframe_split": {
            "columns": ["f1", "f2", "f3", "f4", "f5"],
            "data": [
                [0.5, 1.2, -1.0, 2.1, 0.03]  # single row of data
            ]
        }
    }

    resp = requests.post(url, json=sample_input)
    
    if resp.status_code == 200:
        print("Response status: 200 (OK)")
        print("Model output:", resp.json())
    else:
        print(f"Error {resp.status_code}: {resp.text}")

if __name__ == "__main__":
    main()

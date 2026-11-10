import requests

url = "http://127.0.0.1:5000/api/upload"

file_path = "../dataset/student_placement.csv"

with open(file_path, "rb") as file:

    response = requests.post(
        url,
        files={
            "file": file
        }
    )

print("Status Code:", response.status_code)
print("Response:")
print(response.json())
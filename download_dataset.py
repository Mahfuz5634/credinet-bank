import urllib.request
import os

url = "https://raw.githubusercontent.com/Paliking/ML_examples/master/LoanPrediction/train_u6lujuX_CVtuZ9i.csv"
os.makedirs("data", exist_ok=True)
dest = os.path.join("data", "loan_data.csv")

print(f"Downloading dataset from {url} to {dest}...")
urllib.request.urlretrieve(url, dest)
print(f"Dataset downloaded successfully! File size: {os.path.getsize(dest)} bytes")

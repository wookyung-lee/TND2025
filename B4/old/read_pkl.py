import pickle

# Load object from pickle file
with open("timeseries_data.pkl", "rb") as f:
    data = pickle.load(f)

print(type(data))
print(data)

# T2_times = {key: val['T2'] for key, val in data.items()} # 200k

# print(T2_times)

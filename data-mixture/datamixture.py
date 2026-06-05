from collections import defaultdict

datasets = [
    {"name": "web", "quality": 0.60, "size": 1000},
    {"name": "books", "quality": 0.95, "size": 200},
    {"name": "code", "quality": 0.90, "size": 300},
]

total_weight = 0

for d in datasets:
    d["weight"] = d["quality"] * d["size"]
    total_weight += d["weight"]

print("Suggested Sampling Ratios")

for d in datasets:
    ratio = d["weight"] / total_weight
    print(f"{d['name']}: {ratio:.2%}")

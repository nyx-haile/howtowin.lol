import json
d = {}
with open("clickhouse_array", "r") as f:
    lines = f.readlines()
for i, l in enumerate(lines):
    d[i] = l.split("\t")[1].strip()

print(d)
json.dump(d, open("clickhouse_array.json", "w"))

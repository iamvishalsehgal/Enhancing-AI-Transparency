import csv

# Load nodes
nodes_file = "RAG/Nodes-Rel/1/nodes.csv"
with open(nodes_file, "r", encoding="utf-8") as f:
    nodes = {row[0] for row in csv.reader(f) if row}

# Validate relationships
relationships_file = "RAG/Nodes-Rel/1/relationships.csv"
invalid_relationships = []
with open(relationships_file, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)  # Skip header row
    for i, row in enumerate(reader, start=2):  # Start line count at 2 (after header)
        if len(row) < 3:
            continue
        start_id, _, end_id = row
        if start_id not in nodes or end_id not in nodes:
            invalid_relationships.append((i, row))

# Print results
if invalid_relationships:
    print("❌ Invalid relationships found:")
    for line_num, row in invalid_relationships:
        print(f"Line {line_num}: {row}")
else:
    print("✅ All relationships are valid!")
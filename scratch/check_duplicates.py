with open("datasets/utilities/liquor_11.txt", "r") as f:
    for line_idx, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        parts = line.split(":")
        if len(parts) < 3:
            continue
        items = [int(x) for x in parts[0].split()]
        if len(items) != len(set(items)):
            print(f"Transaction {line_idx} has duplicates: {items}")
            break

import pandas as pd

project_info = pd.read_csv("data/raw/CWE-Bench-Java/raw_data/project_info.csv")
fix_info = pd.read_csv("data/raw/CWE-Bench-Java/raw_data/fix_info.csv")

project_slugs = set(project_info["project_slug"].astype(str).str.strip())
fix_slugs = set(fix_info["project_slug"].astype(str).str.strip())

missing_in_project = sorted(fix_slugs - project_slugs)

print("So slug co trong fix_info nhưng khong co trong project_info:", len(missing_in_project))
print(missing_in_project[:20])
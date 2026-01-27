Prepare Vul4J-style JSONL for CodeBERT fine-tuning (binary classification).

Input JSONL schema (per line):
  { "vul_id", "file", "method", "version", "label", "code" }

Output:
  train.jsonl / valid.jsonl / test.jsonl with fields:
    { "id", "vul_id", "file", "method", "version", "text", "labels", "code_hash", "is_test" }

Key features:
- Optional filter out src/test/
- Lightweight Java comment stripping (optional)
- Truncation / corruption filters (brace balance, suspicious start tokens)
- Dedup by code_hash
- Group split by vul_id to reduce leakage
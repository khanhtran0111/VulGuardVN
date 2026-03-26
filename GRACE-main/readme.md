## About the Models

- The model can be runned from `llmpre.py`:
```contents=format(inputCode)+templates[1]+templates[2]+format(inputnode)+templates[3]+format(inputedge)+templates[4]+format(inputex)```
(node + edge hiện tại không dùng được)

- If you need to run the version with basic prompts, please execute `basep.py`:
```contents=format(inputCode)+templates[1]```

- Please note that you need to fill in the basic API interface and key. Specifically, `GEMINI_MODEL="xxx"`, and `API_KEY="xxx"` or `export GOOGLE_API_KEY="xxx"`

## Figure

We put the figures in `figs\` folder

## Data Processing and Inference Flow

```text
┌─────────────────────────────────────────────────────────────────────┐
│                    RAW DATASET (BigVul / MSR-style)                │
│  Mỗi dòng có thể chứa:                                             │
│  - func_before / func_after                                        │
│  - vul (0/1)                                                       │
│  - CVE ID / CWE ID / Summary / patch / commit_id / project ...     │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                │  (tiền xử lý: chọn code + label cho từng sample)
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FUNCTION-LEVEL SAMPLE                           │
│  Ví dụ một sample sau khi rút gọn còn:                             │
│  - func   : đoạn code hàm cần kiểm tra                             │
│  - target : vulnerable / non-vulnerable                            │
└─────────────────────────────────────────────────────────────────────┘
                 │                                 │
                 │                                 │
                 │                                 │
                 ▼                                 ▼
┌───────────────────────────────┐      ┌──────────────────────────────┐
│  NHÁNH 1: DEMONSTRATION       │      │  NHÁNH 2: GRAPH STRUCTURE    │
│  RETRIEVAL                    │      │                              │
│                               │      │  func                        │
│  func                         │      │   │                          │
│   │                           │      │   ▼                          │
│   ▼                           │      │  Joern                       │
│  CodeT5 embedding             │      │   │                          │
│   │                           │      │   ▼                          │
│   ▼                           │      │  CPG = AST + CFG + PDG       │
│  semantic retrieval           │      │   │                          │
│   │                           │      │   ▼                          │
│   ▼                           │      └──────────────────────────────┘
│  top-k examples               │
│   │                           │
│   ▼                           │
│  rerank lexical + syntactic   │
│   │                           │
│   ▼                           │
│  example tốt nhất             │
└───────────────────────────────┘
                 │                                 │
                 └───────────────┬─────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         PROMPT CHO GPT-4                            │
│                                                                     │
│  [Code snippet]                                                     │
│  [Node information]                                                 │
│  [Edge information]                                                 │
│  [Retrieved example / demonstration]                                │
│                                                                     │
│  => Hỏi LLM: hàm này vulnerable hay non-vulnerable?                 │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           OUTPUT                                    │
│                 vulnerable / non-vulnerable                         │
└─────────────────────────────────────────────────────────────────────┘
```
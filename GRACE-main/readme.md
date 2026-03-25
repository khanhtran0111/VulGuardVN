## About the Models

- The model can be runned from `llmpre.py`:
```contents=format(inputCode)+templates[1]+templates[2]+format(inputnode)+templates[3]+format(inputedge)+templates[4]+format(inputex)```
(node + edge hiện tại không dùng được)

- If you need to run the version with basic prompts, please execute `basep.py`:
```contents=format(inputCode)+templates[1]```

- Please note that you need to fill in the basic API interface and key. Specifically, `GEMINI_MODEL="xxx"`, and `API_KEY="xxx"` or `export GOOGLE_API_KEY="xxx"`

## Figure

We put the figures in `figs\` folder
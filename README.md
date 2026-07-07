# AI360_team19

Install environment: `uv sync`

Load Datasets: https://disk.yandex.ru/d/xVd33nmvuR3NTw

Datasets should be placed like: `./data/Yahoo/...`, `./data/AIOPS/...`

Run benchmark:
```bash
uv run run_benchmark.py \
  --datasets "NAB, TODS, UCR, WSD, Yahoo" \
  --models models.json5
```

See Anomaly Detection Examples: `anomaly_detection_example.ipynb`
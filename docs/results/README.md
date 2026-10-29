# Benchmark results

These CSV files are the numbers behind the report's Results section. They were produced by `benchmark.py` on a 4-core CPU with no GPU.

| File | What it contains |
|---|---|
| `face_counts.csv` | Faces found per test image by each detector, next to the true count |
| `robustness.csv` | Every detector on every image under 12 degradations |
| `robustness_by_condition.csv` | The same, summarised per condition (used for the chart) |
| `summary.csv` | Detection rate, extra boxes and mean time per detector |
| `timing.csv` | Mean and standard deviation of 10 runs for each operation and style |
| `quality.csv` | PSNR and SSIM of each filter against the original |
| `ground_truth.csv` | True face counts (same as `datasets/test_faces/ground_truth.csv`) |

To regenerate them (timings will differ on your machine):

```bash
python benchmark.py --images datasets/test_faces --gt datasets/test_faces/ground_truth.csv
```

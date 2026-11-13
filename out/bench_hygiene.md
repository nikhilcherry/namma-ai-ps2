# Benchmark (x86_64 (not a Pi))

Python 3.13.13, Linux-7.0.11-76070011-generic-x86_64-with-glibc2.39, 30 timed frames per run after 5 warm-up, 24 distinct frames cycled, inference time only. Start state: {'throttled': 'n/a', 'temp': 'n/a'}.

| model | run | mean ms | median ms | p95 ms | FPS (1000/mean) | temp after | throttled after |
|---|---|---|---|---|---|---|---|
| int8h | 1 | 84.7 | 80.6 | 160.3 | 11.81 | n/a | n/a |
| int8h | 2 | 36.5 | 30.1 | 60.8 | 27.37 | n/a | n/a |
| int8 | 1 | 34.0 | 31.7 | 51.6 | 29.43 | n/a | n/a |
| int8 | 2 | 131.7 | 141.0 | 191.9 | 7.59 | n/a | n/a |
| fp32 | 1 | 43.5 | 31.4 | 91.7 | 23.00 | n/a | n/a |
| fp32 | 2 | 71.4 | 69.6 | 112.7 | 14.01 | n/a | n/a |

- **int8h: runs differ by 132%** (> 10%). Throttling or noise: cool the board, close apps, rerun.
- **int8: runs differ by 288%** (> 10%). Throttling or noise: cool the board, close apps, rerun.
- **fp32: runs differ by 64%** (> 10%). Throttling or noise: cool the board, close apps, rerun.

Report mean, median and p95 together. Do not quote the faster run alone.

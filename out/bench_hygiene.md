# Benchmark (x86_64 (not a Pi))

Python 3.13.13, Linux-7.0.11-76070011-generic-x86_64-with-glibc2.39, 40 timed frames per run after 5 warm-up, 30 distinct frames cycled, inference time only. Start state: {'throttled': 'n/a', 'temp': 'n/a'}.

| model | run | mean ms | median ms | p95 ms | FPS (1000/mean) | temp after | throttled after |
|---|---|---|---|---|---|---|---|
| n_int8 | 1 | 118.7 | 119.4 | 170.7 | 8.43 | n/a | n/a |
| n_int8 | 2 | 113.9 | 100.7 | 211.8 | 8.78 | n/a | n/a |
| n_fp32 | 1 | 77.4 | 77.2 | 111.2 | 12.91 | n/a | n/a |
| n_fp32 | 2 | 81.2 | 73.5 | 124.1 | 12.31 | n/a | n/a |

Both runs of every model agree within 10%.

Report mean, median and p95 together. Do not quote the faster run alone.

# Scheduler Comparison

| scheduler | completed_requests | total_steps | total_time_ms | throughput_rps | ttft_p50_ms | ttft_p99_ms | latency_p50_ms | latency_p99_ms | queue_time_p50_ms | tbt_p50_ms | prefill_p50_ms | cache_hits | cache_lookups | cache_hit_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fcfs | 2 | 20000 | 22362.12 | 0.09 | 23.3 | 37.41 | 1457.9 | 2202.9 | 4.25 | 20.04 | 19.05 | 0 | 381 | 0.0 |
| sjf | 33 | 20000 | 35633.62 | 0.93 | 38.55 | 64.62 | 1595.15 | 2586.01 | 9.7 | 20.28 | 26.0 | 0 | 8830 | 0.0 |
| sarathi_serve | 64 | 3023 | 62195.12 | 1.03 | 22789.03 | 44514.67 | 24333.41 | 45974.87 | 22722.83 | 20.0 | 47.3 | 0 | 56910 | 0.0 |
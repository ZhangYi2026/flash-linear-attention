| Workload | Layout | Dtype | Before (us) | After (us) | Latency change |
| --- | --- | --- | ---: | ---: | ---: |
| 1,256,2,64,64 | dense | float16 | 210.296 | 114.898 | -45.36% |
| 1,256,2,64,64 | packed | float16 | 177.905 | 106.531 | -40.12% |
| 1,256,2,64,64 | dense | bfloat16 | 212.835 | 115.895 | -45.55% |
| 1,256,2,64,64 | packed | bfloat16 | 173.578 | 109.765 | -36.76% |

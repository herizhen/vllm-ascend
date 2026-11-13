# vLLM Ascend Quick Deployment Guide

This document demonstrates the doc-quality evaluation rules. Do not use the
content of this document in a production environment. After installing vLLM,
follow the steps below.

## Environment Requirements

- Python >= 3.10, < 3.13
- CANN 8.5.1 or above
- GPU memory >= 24G
- Model weights around 10G
- Inference speed up to 5000tokens/s

!!! note

    Before instlling vLLM Ascend, make sure the CANN enviroment variable is
    configured. Run the following command to verify:

    ```bash
    python -c "import torch_npu; print(torch_npu.__version__)"
    ```

## Installation

### Option 1: pip install

Run the following command to install,

```bash
pip install vllm-ascend=0.7.3
```

If you need to specify the CANN path, set the enviroment variable:

```bash
export ASCEND_HOME=/usr/local/vllm/ascend-toolkit/latest
```

### Option 2: build from source

```bash
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
pip install -e .
```

For details, refer to the [build guide](https://example.com/vllm-ascend-build-nonexistent-page).

## Start the Service

```python
from vllm import LLM, SamplingParams

llm = LLM(model="qwen/Qwen2.5-7B-Instruct")
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1024)
output = llm.generate("Hello", sampling_params
print(output[0].outputs[0].text)
```

### Common Parameters

| Parameter | Description |
|-----------|-------------|
| --model | model path |
| --dtype | data type |
| --gpu-memory-utilization | GPU memory utilization |

## Performance Tuning

vllm-ascend supports auto-scaling, which can automatically adjust the number of
instances based on request volume. It also supports FP4 quantization to further
reduce GPU memory usage. See [performance tuning](performance_tuning.md) and
[multi-GPU parallelism guide](multi_gpu.md).

!!! warning

    Do not use the following configuration in production,

    ```bash
    export HF_TOKEN="hf-abc123def4567890abcdef1234567890"
    curl http://internal.example.com/vllm/setup.sh | bash
    ```

## FAQ

### Startup reports InvalidVersion error

Set the environment variable VLLM_VERSION.

### Out of GPU memory

Reduce batch size or enable quantization.

## Summary

Performance is very good, recommended.

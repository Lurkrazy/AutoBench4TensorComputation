# TVM Meta Schedule Benchmarks

This document provides a comprehensive guide to the TVM Meta Schedule benchmarking capabilities within the AutoBench4TensorComputation framework.

## Overview

TVM Meta Schedule is an automated tensor program optimization system that uses machine learning-guided search to find optimal schedules for tensor computations. This repository provides comprehensive benchmarking tools to evaluate Meta Schedule performance across various workloads and compare it with other tensor computation engines.

### Implementation Comparison

| Feature | autobench/ | autobench-main/ |
|---------|------------|-----------------|
| **Purpose** | Standalone Meta Schedule research | Multi-engine benchmarking |
| **Workloads** | 10 workloads (C1D, C2D, C3D, DIL, DEP, GRP, T2D, CBR, GEMM×2) | 5 workloads (C1D, C2D, C3D, GEMM×2) |
| **Engines** | TVM Meta Schedule only | 6 engines (TVM MS, Ansor, CUTLASS, Triton, Torch, cuBLAS) |
| **Batch Processing** | ✅ Multiple workloads/batch sizes | ✅ Single workload per run |
| **Tuning Focus** | ✅ Research & development | ✅ Production comparison |
| **Script** | `engine_tvm_ms.py` | `bench.py` |

### Supported Engines

- **TVM Meta Schedule** (`tvm_ms`) - ML-guided auto-scheduling
- **TVM Ansor** (`tvm_ansor`) - Template-based auto-scheduling  
- **CUTLASS** (`cutlass`) - NVIDIA's high-performance CUDA library
- **Triton** (`triton`) - GPU kernel language and compiler
- **PyTorch** (`torch`) - PyTorch native implementations
- **NVCUBLAS** (`nvcublas`) - NVIDIA's optimized BLAS library

## Supported Workloads

### Core Convolution Operations
- **C1D**: 1D Convolution (NLC layout)
- **C2D**: 2D Convolution (NHWC layout) 
- **C3D**: 3D Convolution (NDHWC layout)

### Specialized Convolution Operations (autobench/ only)
- **DIL**: Dilated Convolution
- **DEP**: Depthwise Convolution
- **GRP**: Grouped Convolution
- **T2D**: Transpose 2D Convolution

### Fused Operations (autobench/ only)
- **CBR**: Conv2D + BatchNorm + ReLU fusion

### Matrix Multiplication
- **GEMM-1024-1024-1024**: 1024×1024×1024 matrix multiplication
- **GEMM-4096-4096-4096**: 4096×4096×4096 matrix multiplication

### End-to-End Models (configurations available)
- **ResNet-18/50**: Image classification models
- **MobileNet-v2**: Efficient mobile architecture
- **BERT-Large**: Transformer language model
- **ViT**: Vision Transformer

**Note**: The repository contains two Meta Schedule implementations:
- `autobench/` - Standalone engine with extended workload support (DIL, DEP, GRP, T2D, CBR)
- `autobench-main/` - Unified benchmarking interface with core workloads (C1D, C2D, C3D, GEMM)

## Quick Start

### Implementation Overview

This repository provides **two Meta Schedule implementations**:

1. **Standalone Engine** (`autobench/op/engine_tvm_ms.py`)
   - Dedicated Meta Schedule tuning interface
   - Extended workload support (includes DIL, DEP, GRP, T2D, CBR)
   - Batch processing of multiple workloads
   - Research-focused with detailed tuning control

2. **Unified Benchmarking** (`autobench-main/op/bench.py`)
   - Multi-engine comparison platform
   - Core workloads (C1D, C2D, C3D, GEMM)
   - Production-focused with standardized metrics
   - Engine comparison capabilities

### Prerequisites

1. **TVM Installation**: Install TVM with Meta Schedule support
```bash
# Install TVM from source or use pre-built packages
pip install apache-tvm
```

2. **CUDA Environment**: Ensure CUDA toolkit is properly installed
```bash
export PATH=/usr/local/cuda/bin:$PATH
export TVM_TARGET="cuda -arch=sm_80"  # Adjust for your GPU
```

3. **Dependencies**: Install required Python packages
```bash
pip install numpy tabulate
```

### Basic Usage

#### Single Workload Benchmark

```bash
# Run Meta Schedule on 2D Convolution
cd autobench-main/op
python bench.py --workload C2D --engine tvm_ms --batch_size 1 --num_trials 1000
```

#### Compare Multiple Engines

```bash
# Compare Meta Schedule with CUTLASS
python bench.py --workload GEMM-1024-1024-1024 --engine tvm_ms --num_trials 1000
python bench.py --workload GEMM-1024-1024-1024 --engine cutlass
```

#### Precision Control

```bash
# Run with different precision settings
python bench.py --workload C2D --engine tvm_ms \
    --input_dtype f16 --acc_dtype f16 --out_dtype f16 \
    --num_trials 2000
```

## Advanced Configuration

### Meta Schedule Parameters

The Meta Schedule engine supports extensive configuration through `get_search_config()`:

```python
TuneConfig(
    num_trials_per_iter=64,           # Trials per iteration
    max_trials_per_task=2000,         # Max trials per task
    max_trials_global=1000,           # Global trial limit
    search_strategy_config={
        "population_size": 2048,       # Genetic algorithm population
        "init_measured_ratio": 0.2,    # Initial measured ratio
        "init_min_unmeasured": 50,     # Min unmeasured candidates
        "genetic_num_iters": 3,        # Genetic algorithm iterations
        "genetic_mutate_prob": 0.85,   # Mutation probability
        "genetic_max_fail_count": 10,  # Max consecutive failures
        "eps_greedy": 0.05,           # Exploration probability
    }
)
```

### Schedule Rules (Tensor Core Optimized)

The framework uses sophisticated scheduling rules optimized for NVIDIA Tensor Cores:

1. **Multi-Level Tiling**: Hierarchical tiling with structure "SSSRRSRS"
2. **Tensor Core Utilization**: Automatic tensor core instruction selection
3. **Memory Optimization**: Shared memory reuse and cooperative fetching
4. **Thread Mapping**: Optimized block and thread index binding
5. **Vectorization**: Automatic vectorization with configurable extents

### Post-Processing Optimizations

- **Cooperative Fetch Rewriting**: Optimize memory access patterns
- **Unbound Block Rewriting**: Handle dynamic tensor shapes
- **Parallel Vectorize Unroll**: Optimize loop structures
- **Reduction Block Rewriting**: Optimize reduction operations
- **Tensor Core Rewriting**: Generate tensor core instructions
- **GPU Code Verification**: Validate generated CUDA code

## Usage Examples

### 1. Standalone Meta Schedule Engine

Use the dedicated Meta Schedule engine for focused tuning:

```bash
cd autobench/op
python engine_tvm_ms.py -w C2D C3D GEMM-1024-1024-1024 -bs 1 4 -n 5000
```

**Parameters:**
- `-w, --workload`: Space-separated list of workloads
- `-bs, --batch-size`: List of batch sizes to test
- `-n, --num-trials`: Number of tuning trials
- `-t, --target`: TVM target specification
- `--work-dir`: Directory for storing tuning logs

### 2. Unified Benchmarking Interface

Use the unified interface for comparing multiple engines:

```bash
cd autobench-main/op
python bench.py --workload C2D --engine tvm_ms --batch_size 4 \
    --target "cuda -arch=sm_80" --num_trials 3000 \
    --log_dir ./results/
```

### 3. RPC Distributed Tuning

For distributed tuning across multiple GPUs:

```bash
python bench.py --workload GEMM-4096-4096-4096 --engine tvm_ms \
    --rpc --rpc-host localhost --rpc-port 9090 --rpc-key gpu \
    --workers 4 --num_trials 10000
```

### 4. Precision Comparison Study

Compare different precision configurations:

```bash
# FP16 end-to-end
python bench.py --workload CBR --engine tvm_ms \
    --input_dtype f16 --acc_dtype f16 --out_dtype f16

# Mixed precision  
python bench.py --workload CBR --engine tvm_ms \
    --input_dtype f16 --acc_dtype f32 --out_dtype f16
```

## Understanding Results

### Output Directory Structure

```
results/
└── {GPU_NAME}/
    └── workloads/
        └── batch_size_{BS}_{WORKLOAD}_{ENGINE}_input_{IN}_acc_{ACC}_output_{OUT}_trials_{TRIALS}/
            ├── env.txt                    # Environment information
            ├── cost_model.xgb            # Trained cost model
            ├── database_tuning_record.json # Tuning records
            └── database_workload.json     # Workload definitions
```

### Environment Information

Each benchmark records detailed environment information:

```
Name                        Value
--------------------------  -----------------------
GPU                         NVIDIA GeForce RTX 3090
Arch                        Ampere
Compute Capacity            (8, 6)
Current SM Clock (MHz)      1695
Current Memory Clock (MHz)  9501
```

### Performance Metrics

Meta Schedule provides comprehensive performance data:
- **Execution Time**: Actual kernel execution time
- **Memory Throughput**: Achieved memory bandwidth
- **Compute Utilization**: Tensor core and CUDA core usage
- **Schedule Quality**: Tuning convergence metrics

## Workload Configurations

### Default Shapes

The framework includes predefined shapes for consistent benchmarking:

#### Core Workloads (both implementations)
```json
{
    "C1D": [1, 256, 64, 64, 3, 1, 1],              // [N, L, CI, CO, kernel, stride, padding]
    "C2D": [1, 56, 56, 64, 64, 3, 1, 1],           // [N, H, W, CI, CO, kernel, stride, padding]
    "C3D": [1, 16, 56, 56, 64, 64, 3, 1, 1],       // [N, D, H, W, CI, CO, kernel, stride, padding]
    "GEMM-1024-1024-1024": [1, 1024, 1024, 1024],  // [batch, M, K, N]
    "GEMM-4096-4096-4096": [1, 4096, 4096, 4096]   // [batch, M, K, N]
}
```

#### Extended Workloads (autobench/ only)
```json
{
    "DEP": [1, 112, 112, 16, 1, 1, 1, 1, 6],       // Depthwise conv with groups=6
    "DIL": [1, 56, 56, 64, 64, 1, 1, 0, 2],        // Dilated conv with dilation=2
    "GRP": [1, 56, 56, 64, 64, 3, 1, 1, 4],        // Grouped conv with groups=4
    "T2D": [1, 56, 56, 64, 64, 3, 1, 1],           // Transpose 2D convolution
    "CBR": [1, 56, 56, 64, 64, 3, 1, 1]            // Conv + BatchNorm + ReLU
}
```

### Custom Workload Definition

To add custom workloads, extend the configuration files:

**For autobench/ (extended workloads):**
1. Add shape configuration in `autobench/op/configs`
2. Implement workload function in `autobench/op/workload_fp16.py`
3. Register in `CONFIGS_F16` dictionary
4. Add to `WORKLOADS` list in `engine_tvm_ms.py`

**For autobench-main/ (core workloads):**
1. Add shape configuration in `autobench-main/utils/configs`
2. Implement workload function in `autobench-main/utils/workloads_fp16.py`
3. Register in `CONFIGS_F16` dictionary

## Performance Analysis

### Example Results

The repository includes example benchmark results in `autobench-main/results/` directory:

```
results/RTX3090/workloads/
├── batch_size_1_GEMM-1024-1024-1024_cutlass_input_f16_acc_f16_output_f16/
├── batch_size_1_GEMM-1024-1024-1024_tvm_ms_input_f16_acc_f16_output_f16_trials_1000/
└── batch_size_1_GEMM-4096-4096-4096_cutlass_input_f16_acc_f16_output_f16/
```

These results demonstrate Meta Schedule performance comparison with CUTLASS on NVIDIA RTX 3090 for GEMM workloads using FP16 precision.

### Comparing Engines

To systematically compare Meta Schedule with other engines:

```bash
#!/bin/bash
# Comprehensive comparison script
WORKLOADS=("C2D" "GEMM-1024-1024-1024" "CBR")
ENGINES=("tvm_ms" "cutlass" "torch")

for workload in "${WORKLOADS[@]}"; do
    for engine in "${ENGINES[@]}"; do
        echo "Benchmarking $workload with $engine"
        python bench.py --workload $workload --engine $engine \
            --batch_size 1 --num_trials 2000 \
            --log_dir ./comparison_results/
    done
done
```

### Tuning Convergence Analysis

Monitor tuning progress by examining `database_tuning_record.json`:

```python
import json
import matplotlib.pyplot as plt

# Load tuning records
with open('database_tuning_record.json', 'r') as f:
    records = [json.loads(line) for line in f]

# Extract performance over time
costs = [record[0][0] for record in records]  # Execution time
plt.plot(costs)
plt.xlabel('Tuning Iteration')
plt.ylabel('Execution Time (ms)')
plt.title('Meta Schedule Tuning Convergence')
plt.show()
```

## Best Practices

### 1. Tuning Budget Selection

- **Development**: 1000-2000 trials for quick iteration
- **Research**: 5000-10000 trials for thorough exploration  
- **Production**: 20000+ trials for optimal performance

### 2. Target Architecture Specification

Always specify your target GPU architecture:

```bash
export TVM_TARGET="cuda -arch=sm_80 -max_threads_per_block=1024"  # A100/RTX 30x0
export TVM_TARGET="cuda -arch=sm_86 -max_threads_per_block=1024"  # RTX 40x0
export TVM_TARGET="cuda -arch=sm_75 -max_threads_per_block=1024"  # RTX 20x0
```

### 3. Memory Considerations

For large workloads, monitor GPU memory usage:

```bash
# Monitor GPU memory during tuning
nvidia-smi -l 1 &
python bench.py --workload GEMM-4096-4096-4096 --engine tvm_ms --num_trials 5000
```

### 4. Reproducibility

For reproducible results:

```python
# Set random seeds in your environment
export PYTHONHASHSEED=0
export TVM_RANDOM_SEED=0
```

## Troubleshooting

### Common Issues

1. **CUDA Version Mismatch**
   ```bash
   # Verify CUDA installation
   nvcc --version
   python -c "import tvm; print(tvm.cuda().exist)"
   ```

2. **Insufficient GPU Memory**
   ```bash
   # Reduce batch size or workload dimensions
   python bench.py --workload C2D --engine tvm_ms --batch_size 1
   ```

3. **Compilation Errors**
   ```bash
   # Check TVM installation and CUDA paths
   python -c "import tvm; print(tvm.target.cuda().str())"
   ```

### Performance Debugging

Enable detailed logging for performance analysis:

```python
import logging
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)
```

## Contributing

To extend the Meta Schedule benchmarks:

1. **Add New Workloads**: Implement in `workload_fp16.py`
2. **Custom Schedule Rules**: Extend `sch_rules_tensor_core()`
3. **New Post-processors**: Add to `postprocs_tensor_core()`
4. **Evaluation Metrics**: Enhance logging and analysis

## References

- [TVM Meta Schedule Documentation](https://tvm.apache.org/docs/how_to/tune_with_autotir/index.html)
- [NVIDIA Tensor Core Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#tensor-cores)
- [CUTLASS Library](https://github.com/NVIDIA/cutlass)

---

For questions and support, please refer to the main repository documentation or open an issue on GitHub.
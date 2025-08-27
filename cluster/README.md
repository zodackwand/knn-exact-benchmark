# Cluster Deployment Guide

## Professional Workflow: Local → Cluster

### 🎯 Three-Tier Testing Strategy

```
Local Development → Cluster Validation → Full-Scale Benchmarking
    (seconds)           (minutes)            (hours/days)
     N ≤ 10K             N ≤ 1M              N ≤ 100M+
```

## Quick Start

### 1. Local Development
```bash
# Test algorithm locally first
python tools/smoke_algo.py --algo your_algorithm --dataset-key toy_gaussian_N1000_D64_nq100_seed42 --k 10

# Run quick comparison
python bench.py --config configs/tiny_comparison.yaml
```

### 2. Cluster Validation
```bash
# Submit to cluster for validation
./cluster/scripts/submit_job.py --config cluster/configs/validation.yaml --time 30min --mem 32GB

# Monitor job
./cluster/scripts/monitor_jobs.py
```

### 3. Full-Scale Benchmarking
```bash
# Submit large-scale benchmark
./cluster/scripts/submit_job.py --config cluster/configs/full_scale.yaml --time 24h --mem 256GB

# Auto-download results when complete
./cluster/scripts/sync_results.py
```

## Cluster Architecture

### Resource Allocation Strategy
- **Small jobs (N<100K):** 1 CPU, 8GB RAM, 30min
- **Medium jobs (N<1M):** 4 CPU, 32GB RAM, 2h  
- **Large jobs (N<10M):** 8 CPU, 128GB RAM, 12h
- **Massive jobs (N>10M):** 16 CPU, 256GB RAM, 24h+

### Dataset Strategy
- **Local:** Toy datasets (MB scale)
- **Cluster:** Real datasets (GB-TB scale)
  - SIFT100M (100M vectors, 128D) ~50GB
  - Deep100M (100M vectors, 96D) ~38GB  
  - Text embeddings (10M+ vectors, 768D) ~30GB+

## File Organization

```
cluster/
├── README.md                 # This file
├── configs/                  # Cluster-specific configs
│   ├── validation.yaml       # Medium-scale validation
│   ├── full_scale.yaml       # Large-scale benchmarking
│   └── massive_scale.yaml    # Research-grade datasets
├── scripts/                  # Job management
│   ├── submit_job.py         # Submit jobs with resource specs
│   ├── monitor_jobs.py       # Check job status
│   └── sync_results.py       # Download completed results
├── templates/                # SLURM/PBS templates
│   ├── slurm_template.sh     # SLURM job template
│   └── pbs_template.sh       # PBS job template  
└── datasets/                 # Large dataset management
    ├── download_large.py     # Download massive datasets
    └── prepare_splits.py     # Create train/test splits
```

## Best Practices

### 1. Resource Efficiency
- Always test locally first (catch bugs early)
- Use appropriate resource allocation (don't waste cluster time)
- Monitor memory usage (kill jobs that exceed allocation)

### 2. Reproducibility  
- Pin random seeds in configs
- Version control all configs and scripts
- Save environment snapshots (requirements.txt, versions)

### 3. Result Management
- Automatic result downloading when jobs complete
- Organized result storage with timestamps
- Comparison tools for cross-run analysis

### 4. Collaboration
- Shared result repository for team access
- Notification system for completed jobs
- Resource usage tracking and reporting

## Effective Cluster Usage

### Memory Planning (500GB Available)
- **Single large job:** Up to 400GB for massive datasets
- **Parallel jobs:** 4x 100GB jobs or 8x 50GB jobs
- **Mixed workload:** Combine small validation + large benchmark jobs

### Time Management
- **Interactive development:** Use small jobs for debugging
- **Batch processing:** Submit multiple configs overnight
- **Long runs:** 24-48h jobs for comprehensive benchmarks

### Cost Efficiency
- Start small: validate locally → small cluster job → scale up
- Use appropriate resources: don't request 256GB for 1M vectors
- Monitor utilization: kill underperforming jobs early
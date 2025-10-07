#!/usr/bin/env python3
"""
Comprehensive Crossover Analysis Runner
Runs benchmarks across multiple SIFT dataset sizes and logarithmic selectivity levels
"""

import os
import sys
import time
import subprocess
import json
from pathlib import Path
from typing import List, Dict

# Configuration
DATASETS = ['sift10k', 'sift100k', 'sift1m']
SELECTIVITY_LEVELS = [100, 50, 25, 12.5, 6.25, 3.125, 1.56, 0.78]  # Logarithmic scale
CONFIG_DIR = Path("configs/crossover_analysis")
RESULTS_DIR = Path("results")

def create_config_file(dataset: str, selectivity: float) -> Path:
    """Create a specific config file from template."""
    template_path = CONFIG_DIR / f"{dataset}_template.yaml"

    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    # Read template
    with open(template_path, 'r') as f:
        content = f.read()

    # Replace selectivity placeholder
    content = content.replace('{selectivity}', str(selectivity))

    # Create specific config file
    config_filename = f"{dataset}_sel{selectivity:.2f}.yaml"
    config_path = CONFIG_DIR / config_filename

    with open(config_path, 'w') as f:
        f.write(content)

    return config_path

def run_benchmark(config_path: Path, benchmark_num: int, total_benchmarks: int) -> Dict:
    """Run a single benchmark and return metadata."""
    print(f"[{benchmark_num}/{total_benchmarks}] {config_path.name}")
    start_time = time.time()

    try:
        # Run benchmark and show periodic status
        cmd = ["python", "bench.py", "--config", str(config_path)]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Check every 15 seconds and show elapsed time
        last_update = start_time
        while process.poll() is None:
            time.sleep(5)
            current_time = time.time()

            if current_time - last_update >= 15:
                elapsed_min = (current_time - start_time) / 60
                print(f"  Running... {elapsed_min:.1f}min")
                last_update = current_time

        stdout, stderr = process.communicate()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd, stdout, stderr)

        elapsed_time = time.time() - start_time

        # Extract final progress info from stdout
        current_algo = "unknown"
        total_queries = 0

        if stdout:
            lines = stdout.strip().split('\n')
            for line in lines:
                if "[+]" in line and "@" in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        current_algo = parts[1]
                elif "progress" in line and "/" in line:
                    # Extract total queries from last progress line
                    try:
                        progress_part = line.split("progress")[1].split("for")[0].strip()
                        if "/" in progress_part:
                            total_queries = int(progress_part.split("/")[1])
                    except:
                        pass

        algo_info = f" ({current_algo}, {total_queries} queries)" if total_queries > 0 else ""
        print(f"  Completed in {elapsed_time/60:.1f} minutes{algo_info}")

        return {
            'config': str(config_path),
            'status': 'success',
            'duration_minutes': elapsed_time / 60,
            'completed_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }

    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"  Failed after {elapsed_time/60:.1f} minutes: {e}")

        return {
            'config': str(config_path),
            'status': 'failed',
            'duration_minutes': elapsed_time / 60,
            'error': str(e),
            'stderr': e.stderr if hasattr(e, 'stderr') else '',
            'completed_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }

def estimate_total_time():
    """Estimate total benchmark time."""
    total_benchmarks = len(DATASETS) * len(SELECTIVITY_LEVELS)
    print(f"Total benchmarks: {total_benchmarks}")
    return total_benchmarks

def main():
    """Main execution function."""
    print(f"Starting crossover analysis at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    total_benchmarks = estimate_total_time()

    # Create config directory if needed
    CONFIG_DIR.mkdir(exist_ok=True)

    # Track progress
    completed_benchmarks = 0
    results = []
    overall_start = time.time()


    # Run all combinations
    for dataset in DATASETS:
        for selectivity in SELECTIVITY_LEVELS:
            completed_benchmarks += 1

            # Create config file
            try:
                config_path = create_config_file(dataset, selectivity)

                # Run benchmark
                result = run_benchmark(config_path, completed_benchmarks, total_benchmarks)
                results.append(result)

                # Clean up config file
                config_path.unlink()

            except Exception as e:
                print(f"Setup failed for {dataset} @ {selectivity}%: {e}")
                results.append({
                    'config': f"{dataset}_sel{selectivity:.2f}.yaml",
                    'status': 'setup_failed',
                    'error': str(e),
                    'completed_at': time.strftime('%Y-%m-%d %H:%M:%S')
                })

    # Summary
    total_elapsed = time.time() - overall_start
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful

    print(f"\nCrossover analysis complete")
    print(f"Total time: {total_elapsed/3600:.1f} hours")
    print(f"Successful: {successful}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")

    # Save execution log
    log_file = f"crossover_analysis_log_{int(time.time())}.json"
    with open(log_file, 'w') as f:
        json.dump({
            'summary': {
                'total_benchmarks': len(results),
                'successful': successful,
                'failed': failed,
                'total_hours': total_elapsed / 3600,
                'completed_at': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'results': results
        }, f, indent=2)

    print(f"Log saved: {log_file}")

if __name__ == "__main__":
    main()
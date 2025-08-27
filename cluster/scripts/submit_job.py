#!/usr/bin/env python3
"""
Cluster job submission script - works locally for development, submits to cluster in production
"""
import argparse
import os
import sys
import yaml
import subprocess
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


def detect_cluster_system():
    """Detect which cluster system is available"""
    if subprocess.run(['which', 'sbatch'], capture_output=True).returncode == 0:
        return 'slurm'
    elif subprocess.run(['which', 'qsub'], capture_output=True).returncode == 0:
        return 'pbs'
    else:
        return 'local'


def load_config(config_path):
    """Load and validate configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def generate_job_script(config, args):
    """Generate job script based on cluster system"""
    system = detect_cluster_system()
    
    if system == 'local':
        return generate_local_script(config, args)
    elif system == 'slurm':
        return generate_slurm_script(config, args)
    elif system == 'pbs':
        return generate_pbs_script(config, args)


def generate_local_script(config, args):
    """Generate local execution script for development/testing"""
    script = f"""#!/bin/bash
# LOCAL EXECUTION MODE - Development/Testing
echo "Running in local mode..."
echo "Config: {args.config}"
echo "Time limit: {args.time}"
echo "Memory: {args.mem}"
echo ""

# Set up environment
cd {project_root}
source .venv/bin/activate 2>/dev/null || echo "No virtual environment found"

# Run benchmark
python bench.py --config {args.config}

echo ""
echo "Job completed successfully!"
"""
    return script


def generate_slurm_script(config, args):
    """Generate SLURM job script"""
    cluster_config = config.get('cluster', {})
    
    script = f"""#!/bin/bash
#SBATCH --job-name=knnbench
#SBATCH --time={args.time}
#SBATCH --mem={args.mem}
#SBATCH --cpus-per-task={args.cpus}
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

# Load modules (adjust for your cluster)
# module load python/3.9
# module load numpy/1.21

# Set up environment
cd {project_root}
source venv/bin/activate

# Run benchmark
python bench.py --config {args.config}
"""
    return script


def generate_pbs_script(config, args):
    """Generate PBS job script"""
    script = f"""#!/bin/bash
#PBS -N knnbench
#PBS -l walltime={args.time}
#PBS -l mem={args.mem}
#PBS -l ncpus={args.cpus}
#PBS -o job_${{PBS_JOBID}}.out
#PBS -e job_${{PBS_JOBID}}.err

# Set up environment
cd {project_root}
source venv/bin/activate

# Run benchmark
python bench.py --config {args.config}
"""
    return script


def submit_job(script_content, args):
    """Submit or execute job"""
    system = detect_cluster_system()
    
    # Write script to temporary file
    script_path = f"temp_job_{os.getpid()}.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)
    
    try:
        if system == 'local':
            print("🖥️  Executing locally...")
            if args.dry_run:
                print("DRY RUN: Would execute locally")
                print(script_content)
            else:
                result = subprocess.run(['bash', script_path], capture_output=False)
                return result.returncode
                
        elif system == 'slurm':
            print("🚀 Submitting to SLURM...")
            if args.dry_run:
                print("DRY RUN: Would submit with sbatch")
                print(script_content)
            else:
                result = subprocess.run(['sbatch', script_path], capture_output=True, text=True)
                print(result.stdout)
                return result.returncode
                
        elif system == 'pbs':
            print("🚀 Submitting to PBS...")  
            if args.dry_run:
                print("DRY RUN: Would submit with qsub")
                print(script_content)
            else:
                result = subprocess.run(['qsub', script_path], capture_output=True, text=True)
                print(result.stdout)
                return result.returncode
                
    finally:
        # Clean up temporary script
        if os.path.exists(script_path):
            os.remove(script_path)


def main():
    parser = argparse.ArgumentParser(description='Submit KNN benchmark job to cluster')
    parser.add_argument('--config', required=True, help='Configuration file')
    parser.add_argument('--time', default='1h', help='Time limit (e.g., 30min, 2h, 1-00:00:00)')
    parser.add_argument('--mem', default='32GB', help='Memory limit (e.g., 16GB, 128GB)')
    parser.add_argument('--cpus', default=4, type=int, help='Number of CPUs')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without executing')
    parser.add_argument('--local-force', action='store_true', help='Force local execution even if cluster available')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Determine system
    if args.local_force:
        system = 'local'
    else:
        system = detect_cluster_system()
    
    # Generate and submit job
    print(f"📋 Config: {args.config}")
    print(f"⏱️  Time: {args.time}")  
    print(f"💾 Memory: {args.mem}")
    print(f"🖥️  System: {system.upper()}")
    print()
    
    script = generate_job_script(config, args)
    result = submit_job(script, args)
    
    if result == 0:
        print("✅ Job submitted successfully!")
    else:
        print("❌ Job submission failed!")
        return 1


if __name__ == '__main__':
    main()
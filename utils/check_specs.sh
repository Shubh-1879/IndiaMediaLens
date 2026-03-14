#!/bin/bash
#PBS -N Check_Hardware
#PBS -l nodes=1
#PBS -q gpu
#PBS -o specs.log
#PBS -j oe

echo "=== GPU SPECIFICATIONS ==="
nvidia-smi

echo -e "\n=== SYSTEM MEMORY (RAM) ==="
free -h

echo -e "\n=== CPU SPECIFICATIONS ==="
lscpu | grep -E "Model name|Thread|Core|Socket|CPU\(s\):"
EOF

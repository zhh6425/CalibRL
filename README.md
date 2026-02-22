<h1 align="center">Controllable Exploration in Hybrid-Policy RLVR for Multi-Modal Reasoning</h1>

<div>
<br>

<div align="center">

[![Github](https://img.shields.io/badge/Repo-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/zhh6425/CalibRL/tree/main)
[![arXiv](https://img.shields.io/badge/Paper-red?style=for-the-badge&logo=arXiv&logoColor=white&labelColor)](https://openreview.net/forum?id=5wxyCidRsK)

</div>

---
## Installation

You can follow the instruction from [VeRL](https://verl.readthedocs.io/en/latest/start/install.html)

or simply follow what we do:

```
# create env
conda create -n verl python==3.11
conda activate verl

# Make sure you have activated verl conda env
# If you need to run with megatron
bash scripts/install_vllm_sglang_mcore.sh
# Or if you simply need to run with FSDP
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh

# Install
pip install -e .[gpu,math,vllm]
```

## Training

Train the CalibRL with script:
```
bash src/examples/run_qwen25vl7b_virl_mix_policy.sh
```

You can train with your own dataset by modifying:
```
data.train_files=[YOUR_DATA_PARQUET]
```

make sure your dataset contain the expert CoT in ```'solution'``` key

or simply modify ```data.pref_key``` to your own CoT key.

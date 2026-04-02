cd /workspace
python -c "
import torch
import torch_geometric
import rdkit
import transformers
import ogb
import deepspeed
import pysmilesutils
import megatron
import apex

print('✓ torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
print('✓ torch_geometric:', torch_geometric.__version__)
print('✓ rdkit OK')
print('✓ transformers:', transformers.__version__)
print('✓ ogb:', ogb.__version__)
print('✓ deepspeed:', deepspeed.__version__)
print('✓ pysmilesutils OK')
print('✓ megatron OK')
print('✓ apex OK')
print()
print('All dependencies installed successfully!')
"
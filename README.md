# Workshop on GPU and slurm usage

## Initial setup
Clone repository, create conda environment, and install package in editable mode:
```bash
git clone https://github.com/baldassarreFe/rpl-workshop
cd rpl-workshop

conda env create -n workshop --file conda.yaml
conda activate workshop
pip install --editable .
```

## Testing
Install testing packages:
```bash
cd rpl-workshop
pip install --editable '.[test]'
pytest
```

## Training
```bash
python -m workshop.train \
    --runpath "path/to/runs/folder" \
    --datapath "path/to/data/folder" \
    --batch_size 64 \
    --learning_rate .001 \
    --weight_decay .00001 \
    --number_epochs 3 \
    --number_workers 2 \
    --device 'cuda'
```
# GPU and SLURM Workshop

<p align="center">
  <a href="https://docs.conda.io/en/latest/miniconda.html"><img alt="Miniconda" src="https://img.shields.io/badge/-Conda-brightgreen?logo=Anaconda&logoColor=white"></a>
  <a href="https://docs.google.com/presentation/d/1mFh92Kwmsc6Cm0RXQH_WxfT7-FtSTtiB7ZWanQ5Gvwg"><img alt="Slides" src="https://img.shields.io/badge/-Slides-yellow?logo=Read-the-docs&logoColor=white"></a>
  <a href="https://gits-15.sys.kth.se/rpl-gpus/wiki/wiki"><img alt="Wiki" src="https://img.shields.io/badge/-Wiki-blue?logo=github"></a>
</p>
<p align="center"><em>KTH - Royal Institute of Technology</em></p>

## Initial setup
Clone repository, create conda environment, and install package in editable mode:
```bash
cd ~
git clone https://github.com/baldassarreFe/rpl-workshop
cd rpl-workshop

conda env create -n workshop --file conda.yaml
conda activate workshop
pip install --editable .
```

## Plain training
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

## Slurm commands

Single job:
```bash
sbatch slurm/single_job.sbatch
```

Grid search using environment variables:
```bash
./slurm/grid_search.sh
```

Grid search using job arrays:
```bash
./slurm/grid_search.sh
```

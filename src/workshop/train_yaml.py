import pyaml
import argparse
from pathlib import Path

from .train import main

def parse_yaml():
    # Just convert the yaml file into an argparse namespace.
    # Ideally one would use the yaml dict directly in the training code.
    
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_config", type=Path)
    args = parser.parse_args()
    with open(args.yaml_config) as f:
        config = pyaml.safe_load(f)
    
    return argparse.Namespace(
        runpath=Path(config["paths"]["runs"]),
        datapath=Path(config["paths"]["data"]),
        batch_size=config["dataloader"]["batch_size"],
        learning_rate=config["optimizer"]["learning_rate"],
        weight_decay=config["optimizer"]["weight_decay"],
        number_epochs=config["optimizer"]["number_epochs"],
        number_workers=config["dataloader"]["number_workers"],
        device=config["session"].get("device", "cpu")
    )
    

if __name__ == '__main__':
    args = parse_yaml()
    main(args)
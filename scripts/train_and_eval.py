## Copyright (C) 2023 Philipp Benner

import ast
import os
import sys, getopt
import torch

from monty.serialization import loadfn, dumpfn

from coordinationnet import CoordinationFeaturesData
from coordinationnet import GraphCoordinationNet, GraphCoordinationNetConfig

### Model configuration
### ---------------------------------------------------------------------------

model_config = GraphCoordinationNetConfig(
    dim_element=200,
    dim_oxidation=10,
    dim_geometry=10,
    dim_csm=128,
    dim_distance=128,
    dim_angle=128,
    bins_csm=20,
    bins_distance=20,
    bins_angle=20,
    distances=True,
    angles=True,
)

## ----------------------------------------------------------------------------

get_model = lambda devices: GraphCoordinationNet(
    # Model components
    model_config=model_config,
    # Dense layer options
    layers=[1024, 512, 128, 64, 1],
    dropout=0.0,
    skip_connections=False,
    batchnorm=False,
    # Data options
    batch_size=64,
    val_size=0.1,
    # Optimizer options
    optimizer="AdamW",
    weight_decay=0.05,
    scheduler="plateau",
    patience_sd=5,
    patience_es=50,
    lr=1e-3,
    num_workers=2,
    # Other
    default_root_dir=f"checkpoints",
    devices=devices,
)

## ----------------------------------------------------------------------------


class MPData(CoordinationFeaturesData):
    def __init__(self, filename):
        X, y = loadfn(filename)

        super().__init__(X, y=y)


## ----------------------------------------------------------------------------


def train_and_eval(inputfile, outputfile, devices):
    # Get a fresh new model
    model = get_model(devices)

    # Split data into train and test data
    data_train, data_test = torch.utils.data.random_split(
        MPData(inputfile), [0.8, 0.2], generator=torch.Generator().manual_seed(42)
    )

    # Optimize parameters on train data
    model.train(data_train)

    # Evaluate model on test data
    y, y_hat, _ = model.test(data_test)

    mae = model.lit_model.loss(y_hat, y).item()

    # Save result
    dumpfn({"y_hat": y_hat.tolist(), "y": y.tolist(), "mae": mae}, outputfile)


## ----------------------------------------------------------------------------


def main():
    inputfile = ""
    outputfile = ""
    devices = [0]

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:o:", ["devices="])
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print("test.py -i <input_file> -o <output_file>")
            sys.exit()
        elif opt == "-i":
            inputfile = arg
        elif opt == "-o":
            outputfile = arg
        elif opt == "--devices":
            devices = ast.literal_eval(arg)
        else:
            assert False, "unhandled option"

    if inputfile == "":
        raise ValueError("input file not specified")
    if outputfile == "":
        raise ValueError("output file not specified")

    print(f"{outputfile}: Training and evaluating model on {inputfile}")

    train_and_eval(inputfile, outputfile, devices)


### ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()

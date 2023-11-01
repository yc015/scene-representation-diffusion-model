import argparse

def getConfig():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True, type=str, help="Name of the output directory")
    parser.add_argument('--weights_type', default="", type=str, help="Which probe weights to use for intervention")

    cfg = parser.parse_args()

    return cfg
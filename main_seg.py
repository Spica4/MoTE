"""
Main entry point for 3D medical image segmentation with incremental learning.

Usage:
    python main_seg.py --config ./exps/medical/twostage_swin_unetr.json
"""

import json
import argparse
from trainer_seg import train


def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict
    args.update(param)  # Add parameters from json

    train(args)


def load_json(setting_path):
    with open(setting_path) as data_file:
        param = json.load(data_file)
    return param


def setup_parser():
    parser = argparse.ArgumentParser(
        description='3D Medical Image Segmentation with Incremental Learning'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='./exps/medical/twostage_swin_unetr.json',
        help='Json file of settings.'
    )
    return parser


if __name__ == '__main__':
    main()

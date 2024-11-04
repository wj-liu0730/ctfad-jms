import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omegaconf import OmegaConf
import argparse

from job.train_yolo import AllConfig as YoloWelding
from job.train_ae import AllConfig as CnnRecon
from job.train_resnet import AllConfig as Classification_ResNet
from job.train_transformer import AllConfig as Classification_Transformer
from job.evaluate_ctfad import AllConfig as Ctfad


ALL_JOBS = {
    "YoloWelding": YoloWelding,
    "CnnRecon": CnnRecon,
    "Classification_ResNet": Classification_ResNet,
    "Classification_Transformer": Classification_Transformer,
    "Ctfad": Ctfad
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--list', action='store_true', help='Listing all available jobs')
    parser.add_argument('--generate', default='', help='Generate a config file for a specific job')

    args = parser.parse_args()

    if args.list:
        print("Available jobs:")
        for job_name in ALL_JOBS.keys():
            print(f"\t{job_name}")
        exit(0)

    if args.generate != '':
        config_directory = "config"
        if not os.path.exists(config_directory):
            os.makedirs(config_directory)

        if args.generate not in ALL_JOBS:
            print(f"Job {args.generate} not found.")
            exit(1)
        OmegaConf.save(ALL_JOBS[args.generate], config_directory + f'/{args.generate}_Config_Template.yaml')
        print(f"Config file generated at {config_directory}/{args.generate}_Config_Template.yaml")
        exit(0)

    print("Please specify the job to generate configurations --list to list all available jobs.")
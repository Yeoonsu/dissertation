import argparse
from data.preprocess import SETGENERATOR
import warnings
warnings.filterwarnings("ignore")
import os
from utils.config import LOGTYPE, COL, META, TASK, INJM
from operator import itemgetter
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from models.model import MODEL, Train, TestClassification, TestRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
from itertools import product
import json

random.seed(42)
torch.manual_seed(42)

# Set device
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define fixed training files
#TEST_FILES = ["pub-TEST-CLEAN.csv"]
TEST_FILES = ["credit-TEST-CLEAN.csv"]
#TEST_FILES = ["BPIC11_f1-TEST-CLEAN.csv"]

# Argument parser

parser = argparse.ArgumentParser(description="Dataset Generator")
parser.add_argument("--dataset_path", type=str, default='data/datasets/', help="data location")
parser.add_argument("--trials", type=int, default=4, help="test trials")
parser.add_argument("--task", required=True, type=str, choices=["next_activity", "event_remaining_time", "outcome", "case_remaining_time"], help="task name")
parser.add_argument("--modelpath", type=str, default='models/model.pt', help="model location")
parser.add_argument("--batchsize", type=int, default=16, help="batchsize for input")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--hiddendim", type=int, default=128, help="size of hidden layer")
parser.add_argument("--embdim", type=int, default=32, help="size of embedding layer")
parser.add_argument("--epoch", type=int, default=300, help="number of epoch")
parser.add_argument("--result_path", type=str, default='result/', help="save location")
args = parser.parse_args()

# Metrics functions for classification and regression
def metrics_class(group):
    if len(group) == 0:
        return pd.Series({'accuracy': 0, 'fscore': 0})
    accuracy = accuracy_score(group['true'], group['pred'])
    fscore = f1_score(group['true'], group['pred'], average='macro')
    return pd.Series({'accuracy': accuracy, 'fscore': fscore})

def metrics_reg(group):
    if len(group) == 0:
        return pd.Series({'mse': 0, 'mae': 0, 'r2': 0, 'rmse': 0})
    mse = mean_squared_error(group['true'], group['pred'])
    mae = mean_absolute_error(group['true'], group['pred'])
    r2 = r2_score(group['true'], group['pred'])
    rmse = np.sqrt(mse)
    return pd.Series({'mse': mse, 'mae': mae, 'r2': r2, 'rmse': rmse})

# Main execution
if __name__ == "__main__":
    # Correcting file roles: test_files -> TRAIN_FILES and vice versa
    train_files = [file for file in os.listdir(args.dataset_path) if file not in TEST_FILES and file.endswith(".csv")]

    for test_file in TEST_FILES:
        for train_file in train_files:
            print(f"Training on: {train_file} | Testing on: {test_file}")

            # Set up train and test paths
            train_csv = os.path.join(args.dataset_path, train_file)
            test_csv = os.path.join(args.dataset_path, test_file)

            true_bin, pred_bin, length_bin, ratio_bin = [], [], [], []

            for rep in range(args.trials):
                print(f"Replication #: {rep}")

                # Generate datasets
                set_generator = SETGENERATOR(train_csv, test_csv, "CLEAN", "CLEAN", args.task, args.batchsize)
                train_loader, test_loader, meta = set_generator.SetGenerator()
                output_dim, num_act, max_length, attr_size, scaler = itemgetter(META.OUTDIM.value, META.NUMACT.value, META.MAXLEN.value, META.ATTRSZ.value, META.SCALER.value)(meta)

                # Initialize model and optimizer
                model = MODEL(num_act, args.embdim, args.hiddendim, 2, attr_size, output_dim, args.task).to(device)
                optimizer = optim.Adam(model.parameters(), lr=args.lr)
                if args.task == TASK.NAP.value:
                    criterion = nn.CrossEntropyLoss()
                elif args.task == TASK.OP.value:
                    criterion = nn.BCEWithLogitsLoss()
                else:
                    criterion = nn.SmoothL1Loss()

                # Train model
                Train(model, train_loader, criterion, optimizer, args.epoch, args.modelpath, args.task)
                torch.cuda.empty_cache()

                # Load the best model and test
                model.load_state_dict(torch.load(args.modelpath))
                if args.task in [TASK.NAP.value, TASK.OP.value]:
                    true, pred, length, ratio = TestClassification(model, test_loader, output_dim, args.task, "CLEAN")
                else:
                    true, pred, length, ratio = TestRegression(model, test_loader, scaler, "CLEAN")

                torch.cuda.empty_cache()
                true_bin.extend(true)
                pred_bin.extend(pred)
                length_bin.extend(length)
                ratio_bin.extend(ratio)

            # Calculate and save results
            if args.task in [TASK.NAP.value, TASK.OP.value]:
                overall_accuracy = accuracy_score(true_bin, pred_bin)
                overall_fscore = f1_score(true_bin, pred_bin, average='macro')

                results_summary = {'overall_accuracy': float(overall_accuracy), 'overall_fscore': float(overall_fscore)}
                with open(f'{args.result_path}{train_file}-{test_file}-overall.json', 'w') as f:
                    json.dump(results_summary, f)

            else:
                overall_mse = mean_squared_error(true_bin, pred_bin)
                overall_mae = mean_absolute_error(true_bin, pred_bin)
                overall_r2 = r2_score(true_bin, pred_bin)
                overall_rmse = np.sqrt(overall_mse)

                results_summary = {'overall_mse': float(overall_mse), 'overall_mae': float(overall_mae), 'overall_r2': float(overall_r2), 'overall_rmse': float(overall_rmse)}
                with open(f'{args.result_path}{train_file}-{test_file}-overall.json', 'w') as f:
                    json.dump(results_summary, f)
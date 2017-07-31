# this parser is written by Justin Su (myself)

import argparse
import re


def main(args):
    pattern = re.compile(r'.*\((\d+) / (\d+)\).*\((\d+) / (\d+)\).*')
    total_tp = .0
    total_tp_fn = .0
    total_tp_fp = .0
    with open(args.input) as f:
        for line in f:
            tp, tp_fn, tp, tp_fp = re.findall(pattern, line)[0]
            total_tp += int(tp)
            total_tp_fn += int(tp_fn)
            total_tp_fp += int(tp_fp)
    precision = total_tp/total_tp_fp
    recall = total_tp/total_tp_fn
    f1 = 2 * precision * recall / (precision + recall)
    print("Precision: {0}\tRecall: {1}\tF1: {2}".format(precision, recall, f1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parses a results file from the pipeline and add up all the scores.")
    parser.add_argument('-i', '--input', help="path to the results file.")
    parser.set_defaults(func=main)
    args = parser.parse_args()
    args.func(args)

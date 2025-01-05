from os.path import basename

import pandas as pd
import os
import re
import csv


def generate_multiView_pairlst():
    views = ['full', 'back', 'front', 'side', 'flat', 'additional']

    pairs_file_train = pd.read_csv(pairLst)
    size = len(pairs_file_train)
    # pairs = []
    print('Loading data pairs ...')

    with open(multiView_pairlst, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["from", "to", "from2"])
        for i in range(size):
            index = pairs_file_train.iloc[i]['from'].rfind('_')
            basename = pairs_file_train.iloc[i]['from'][:index]

            pair = [pairs_file_train.iloc[i]['from'], pairs_file_train.iloc[i]['to']]
            pair_from = os.path.splitext(pairs_file_train.iloc[i]['from'])[0].split('_')[-1]
            pair_from = re.sub(r'\d+', '', pair_from)

            pair_to = os.path.splitext(pairs_file_train.iloc[i]['to'])[0].split('_')[-1]
            pair_to = re.sub(r'\d+', '', pair_to)

            found = False
            for view in views:
                if view == pair_from or view == pair_to:
                    continue

                pattern = basename + '_' + '\d*' + view + '.jpg'

                files = os.listdir(data_root)
                matches = [file for file in files if re.match(pattern, file)]
                if len(matches) != 0:
                    pair.append(matches[0])
                    found = True
                    writer.writerow(pair)
                    break
            if not found:
                print(pairs_file_train.iloc[i]['from'],pairs_file_train.iloc[i]['to'])
                pair.append(pairs_file_train.iloc[i]['from'])
                writer.writerow(pair)


# 101967->101689
if __name__ == '__main__':
    data_root = '../dataset/fashion/train'
    pairLst = '../dataset/fashion/fashion-resize-pairs-train.csv'
    multiView_pairlst = '../dataset/fashion/fashion-resize-multiView-pairs-train.csv'
    generate_multiView_pairlst()

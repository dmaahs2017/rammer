#!/usr/bin/env python3
import glob
import shutil
import os


def main():
    ham_files = glob.glob('**/ham/*', recursive=True)
    spam_files = glob.glob('**/spam/*', recursive=True)
    summary_files = glob.glob('**/Summary.txt', recursive=True)

    ham_split = int(len(ham_files) * .8)
    spam_split = int(len(spam_files) * .8)

    train_ham = ham_files[:ham_split]
    validation_ham = ham_files[ham_split:]

    train_spam = spam_files[:spam_split]
    validation_spam = spam_files[spam_split:]

    i = 0
    for file in train_ham:
        shutil.move(file, f"data/train/ham/{i}.txt")
        i += 1
    for file in validation_ham:
        shutil.move(file, f'data/validate/ham/{i}.txt')
        i += 1
    for file in train_spam:
        shutil.move(file, f'data/train/spam/{i}.txt')
        i += 1
    for file in validation_spam:
        shutil.move(file, f'data/validate/spam/{i}.txt')
        i += 1




if __name__ == '__main__':
    main()

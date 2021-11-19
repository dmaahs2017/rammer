#!/bin/bash
# use this script in the directory you want the files downloaded
curl -OL http://www.aueb.gr/users/ion/data/enron-spam/preprocessed/enron1.tar.gz -o data/raw/enron1.tar.gz &
curl -OL http://www.aueb.gr/users/ion/data/enron-spam/preprocessed/enron2.tar.gz -o data/raw/enron2.tar.gz &
curl -OL http://www.aueb.gr/users/ion/data/enron-spam/preprocessed/enron3.tar.gz -o data/raw/enron3.tar.gz &
curl -OL http://www.aueb.gr/users/ion/data/enron-spam/preprocessed/enron4.tar.gz -o data/raw/enron4.tar.gz &
curl -OL http://www.aueb.gr/users/ion/data/enron-spam/preprocessed/enron5.tar.gz -o data/raw/enron5.tar.gz &
curl -OL http://www.aueb.gr/users/ion/data/enron-spam/preprocessed/enron6.tar.gz -o data/raw/enron6.tar.gz &

__author__ = "Lech Szymanski"
__organization__ = "COSC343/AIML402, University of Otago"
__email__ = "lech.szymanski@otago.ac.nz"
__date__ = "August 2022"

import numpy as np
import os
import re
import pickle
import gzip
import tarfile
import sys




class dataset_trec:

    def __init__(self, N=None):

        picklefile = 'trec07p.pickle'
        if not os.path.exists(picklefile):
            sys.stdout.write('Reading data for the first time...')
            sys.stdout.flush()
            datadir = 'trec07p'
            if not os.path.exists(datadir):
                filepath = 'trec07p.tgz'
                if not os.path.exists(filepath):
                    raise FileNotFoundError('trec07p.tgz not found in the project directory.')

                tarfile.open(filepath, 'r:gz').extractall('./')

            subjects, subject_words, labels = self.read_from_dir(dir_name=datadir)

            # Save and archive the model
            with gzip.open(picklefile, 'w') as f:
                pickle.dump((subjects, subject_words,labels), f)
            sys.stdout.write('done\n')
            sys.stdout.flush()

        else:
            with gzip.open(picklefile) as f:
                (subjects, subject_words,labels) = pickle.load(f)

        self.subjects = subjects
        self.subject_words = subject_words
        self.labels = labels

    def subject_to_word_list(self,subject):
        wordlist = []
        subject_text = subject.partition('Subject: ')[-1]
        wordstr = subject_text.rstrip()
        for words in re.split(' |/|-|\(|\)|\?|\[|\]|=|\.|\:\'', wordstr):
            if len(words) == 0:
                continue

            skipWord = False
            while len(words) > 0:
                if words[-1] == ',':
                    words = words[:-1]
                elif words[-1] == '@':
                    words = words[:-1]
                elif words[-1] == ';':
                    words = words[:-1]
                elif words == '[R]' or words == 'Re:' or words == 'Fwd:' or words == '&' or words == '/' or words == 'RE':
                    skipWord = True
                    break
                elif words[-1] == '"':
                    words = words[:-1]
                elif words[-1] == ')':
                    words = words[:-1]
                elif words[0] == '"':
                    words = words[1:]
                elif words[0] == '(':
                    words = words[1:]
                elif words[0] == '#':
                    words = words[1:]
                elif words[-1] == '.' or words[-1] == '?' or words[-1] == ':' or words[-1] == '!':
                    words = words[:-1]
                else:
                    break

            if skipWord or len(words) == 0:
                continue

            wordlist.append(words.lower())
        return wordlist

    def read_from_dir(self,dir_name):

        label_file = os.path.join(dir_name, 'full','index')

        with open(label_file) as f:
           label_strings = f.readlines()

        labels = []
        subject_words = []
        subjects = []

        for label_s in label_strings:
            match = re.search('inmail.(\d+)', label_s)
            if match:
                l = label_s.split()[0]
                data_file = os.path.join(dir_name,'data',match.group(0))

                wordlist = []
                with open(data_file,encoding="utf8", errors='ignore') as f:
                    firstLine = False

                    for line in f:
                        if firstLine:
                            line = line.split()
                            if line[0] == 'From':
                                wordlist.append(line[1])
                            else:
                                print("What gives?")
                            firstLine = False
                            continue

                        match = re.search('(?:Subject).*', line)
                        if match:
                            subject = match.group(0)
                            wordlist = self.subject_to_word_list(subject)

                            if len(wordlist) > 0:
                                subjects.append(subject)
                                subject_words.append(wordlist)
                                labels.append(l)

                            break


        return subjects, subject_words, labels

    @staticmethod
    def load(N=None):
        # Read the data
        return dataset_trec(N=N)






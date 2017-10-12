# STUDENTS: DON'T RUN THIS FILE. WE HAVE CREATED CLOZE FOR YOU.

import collections
import numpy as np
import numpy.random as random
import sys

BLANK = "<blank>"  # special token for blanks
SHORT_THRES = 10  # length threshold for a sentence to be considered short and thus be removed
LONG_THRES = 30  # length threshold for a sentence to be considered long and thus has max number of blanks
MAX_BLANK_FREQENCY_RANK = 500  # maximum frequency rank of the word type to be used to create blank
STOP_LIST = [".", ",", "(", ")", "!", "#", "%", "@-@", "&quot;", "-", ";", ":"]

if len(sys.argv) < 4:
  sys.stderr.write("usage: python create_cloze.py [test_file] [train_file] [max_blanks_per_sentence]\n")
  sys.exit(1)
else:
  test_file = open(sys.argv[1], 'r')
  train_file = open(sys.argv[2], 'r')
  cloze_question = open(sys.argv[1] + ".cloze", 'w')
  cloze_answer = open(sys.argv[1] + ".answer", 'w')
  max_blanks_per_sent = int(sys.argv[3])

# first pass: collect count
token_cnt = collections.Counter()
for line in train_file:
  tokens = line.split()
  for token in tokens:
    token_cnt[token] += 1

wordlist = token_cnt.most_common(n=MAX_BLANK_FREQENCY_RANK)
wordlist = [pair[0] for pair in wordlist]

for line in test_file:
  tokens = line.strip().split()
  if len(tokens) < SHORT_THRES:
    continue
  candidate_indexes = []
  for idx, token in enumerate(tokens):
    if token in wordlist and token not in STOP_LIST:
      candidate_indexes.append(idx)

  blank_indexes = []
  for blank in range(int(min(np.ceil(max_blanks_per_sent * len(candidate_indexes) / LONG_THRES), 3))):
    idx = random.randint(len(candidate_indexes))
    blank_indexes.append(candidate_indexes[idx])
    del candidate_indexes[idx]
  blank_indexes = sorted(blank_indexes)

  answers = []
  for idx in blank_indexes:
    answers.append(tokens[idx])
    tokens[idx] = BLANK
  cloze_question.write(" ".join(tokens) + "\n")
  cloze_answer.write(" ".join(answers) + "\n")

test_file.close()
cloze_question.close()
cloze_answer.close()

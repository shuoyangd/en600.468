import optparse
import os
import datetime

## Assignment info ##############################################
#
# All four values must be defined

# The assignment's name
name = 'Cloze'

# Text used in the leaderboard column header
scoring_method = 'Accuracy'

# Set to true if highest scores are best
reverse_order = True

# The deadline YYYY, MM, DD, HH, MM (24 hour format)
deadline = datetime.datetime(2017, 11, 02, 23, 59)

#################################################################

answer_file = "%s/cloze_data/test.en.txt.answer" % os.path.dirname(os.path.realpath(__file__))

def oracle():
  return float(1.0)

def score(pred_input, assignment_key, test = False):
  right = 0
  total = 0
  answers = open(answer_file)
  answer_lines = answers.readlines()
  pred_lines = pred_input.split('\n')[:-1]
  assert len(answer_lines) == len(pred_lines)
  for answer_line, pred_line in zip(answer_lines, pred_lines):
    answer_tokens = answer_line.split()
    pred_tokens = pred_line.split()
    assert len(answer_tokens) == len(pred_tokens)
    for answer, pred in zip(answer_tokens, pred_tokens):
      if answer == pred:
        right += 1
      total += 1
  return float(right) / float(total), 100

if __name__ == "__main__":
  optparser = optparse.OptionParser()
  optparser.add_option("-a", "--answer", default="cloze_data/test.en.txt.answer", help="answer filename")
  optparser.add_option("-p", "--pred", help="prediction filename")
  opts, args = optparser.parse_args()

  answer_file = opts.answer
  print(score(open(opts.pred).read(), 4))

# Part-of-Speech-Tagging
Implemented a program to predict Part of Speech tags for untagged text by performing inference using a Hidden Markov Model I created with an initial, transition and observation probability tables.

**Overview**
Natural Language Processing (NLP) is a subset of AI that focuses on the understanding and
generation of written and spoken language. This involves a series of tasks from low-level speech
recognition on audio signals to high-level semantic understanding and inferencing on the parsed
sentences.

One task within this spectrum is Part-Of-Speech (POS) tagging. Every word and punctuation symbol
is understood to have a syntactic role in its sentence, such as nouns (denoting people, places or
things), verbs (denoting actions), adjectives (which describe nouns) and adverbs (which describe
verbs), to name a few. Each word in a text is therefore associated with a part-of-speech tag (usually
assigned by hand), where the total number of tags can depend on the organization tagging the text.

**Testing**
Tested my program using several training and test file combinations. Each command specifies one or more training files 
(separated by spaces), one test file, and one output file.

**Training, Test and Output File Formats**
Each training file contains text and POS tags. Each line has one word/punctuation, a colon,
and the POS tag. I have considered the POS tags in the training files "correct" or "ground-truth."

eg. 
Detective : NP0
Chief : NP0
Inspector : NP0
John : NP0
McLeish : NP0

Each test file contains the text only without the POS tags.

eg. 
Detective
Chief
Inspector

POS tags in the output file are predicted by your HMM model instead of the "ground-truth" POS tags from the training files.

**Constraints**
Program must terminate within 5 minutes to earn marks for each test case.

**Effectiveness**
Certain training files have removed the POS tags to be used for testing purposes. The POS tags are therefore know for these files
and the effectiveness of the program will be the total number of correctly predicted POS tags divided by the total number of words
in the testing file.

**Project Result**
Accuracy: POS tags for a testing file with no repetitive sentences with training files were able to correctly predict the POS tags 87.3% of the time.
Time: Able to process 400000 POS tags from training files and predict 50000 POS tags all in under 3 minutes.

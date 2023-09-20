import os
import sys
import argparse
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from copy import deepcopy
import time

period = '.'
TAGS = ["AJ0", "AJC", "AJS", "AT0", "AV0", "AVP", "AVQ", "CJC", "CJS", "CJT", "CRD",
        "DPS", "DT0", "DTQ", "EX0", "ITJ", "NN0", "NN1", "NN2", "NP0", "ORD", "PNI",
        "PNP", "PNQ", "PNX", "POS", "PRF", "PRP", "PUL", "PUN", "PUQ", "PUR", "TO0",
        "UNC", 'VBB', 'VBD', 'VBG', 'VBI', 'VBN', 'VBZ', 'VDB', 'VDD', 'VDG', 'VDI',
        'VDN', 'VDZ', 'VHB', 'VHD', 'VHG', 'VHI', 'VHN', 'VHZ', 'VM0', 'VVB', 'VVD',
        'VVG', 'VVI', 'VVN', 'VVZ', 'XX0', 'ZZ0', 'AJ0-AV0', 'AJ0-VVN', 'AJ0-VVD',
        'AJ0-NN1', 'AJ0-VVG', 'AVP-PRP', 'AVQ-CJS', 'CJS-PRP', 'CJT-DT0', 'CRD-PNI', 'NN1-NP0', 'NN1-VVB',
        'NN1-VVG', 'NN2-VVZ', 'VVD-VVN', 'AV0-AJ0', 'VVN-AJ0', 'VVD-AJ0', 'NN1-AJ0', 'VVG-AJ0', 'PRP-AVP',
        'CJS-AVQ', 'PRP-CJS', 'DT0-CJT', 'PNI-CRD', 'NP0-NN1', 'VVB-NN1', 'VVG-NN1', 'VVZ-NN2', 'VVN-VVD']
num_tags = 91


class HMM:
    """
    Class for generating prediction tables
    """

    def __init__(self, input_filenames):
        """
        :param input_filename: String that is name of document that holds the original board position
        :type input_filename: String
        """
        self.initial_prob_table = np.zeros(91)
        self.transition_prob_table = np.zeros((91,91))
        self.observation_prob_table = np.zeros((91,1))
        self.words = {}
        prev_POS = None
        for training_file in input_filenames:
          file = open(training_file, "r")
          end_sentence = True
          for line in file:
            prev_POS, end_sentence = self.word_processor(line, end_sentence, prev_POS)
        self.observation_prob_table = np.delete(self.observation_prob_table,0,1)
        self.predictions = []
        self.initial_prob_table = self.initial_prob_table/np.sum(self.initial_prob_table)
        self.transition_prob_table = self.transition_prob_table/np.sum(self.transition_prob_table, axis = 0)
        self.transition_prob_table[np.isnan(self.transition_prob_table)] = 0
        self.observation_prob_table = self.observation_prob_table/np.sum(self.observation_prob_table, axis = 0)
        self.observation_prob_table[np.isnan(self.observation_prob_table)] = 0

    
    def word_processor(self, line, start_of_sentence, prev_POS):
      """
        Function to process every line in training file to update prediciton models
        Returns True if end of sentence detected (eg. period), else otherwise
        :param input_filename: String that is name of document that holds the original board position
        :type input_filename: String
      """
      index = line.index(" : ")
      word = line[:index].lower()
      word_ind = self.words.get(word)
      if line[-1] != '\n':
        POS = line[index+3:]
      else:
        POS = line[index+3:-1]
      end_sentence = False

      if start_of_sentence:
        self.initial_prob_table[TAGS.index(POS)] += 1
      else:
        self.transition_prob_table[TAGS.index(prev_POS),TAGS.index(POS)] += 1
      
      if word_ind == None:
        self.words[word] = self.observation_prob_table.shape[1]-1
        self.observation_prob_table = np.hstack((self.observation_prob_table, np.zeros((91,1))))
      
      self.observation_prob_table[TAGS.index(POS), self.words.get(word)+1] += 1

      if word == period:
        end_sentence = True
      
      return POS, end_sentence 
    
    def predictor(self, test_file, output_filename, starting_time):
      file = open(test_file, "r")
      current_sentence = []
      for line in file:
        if time.time() - starting_time > 290:
          break
        if line[-1] == "\n":
          word = line.rstrip()
        else:
          word = line
        current_sentence.append(word)
        if word == period:
          probs, prev = self.viterbi(deepcopy(current_sentence))
          POS_list = self.backtrack(probs, prev, len(current_sentence))
          for i in range(len(current_sentence)):
            self.predictions.append((current_sentence[i], TAGS[POS_list[i]]))
          current_sentence = []
      if current_sentence != []:
        probs, prev = self.viterbi(current_sentence)
        POS_list = self.backtrack(probs, prev, len(current_sentence))
        for i in range(len(current_sentence)):
          self.predictions.append((current_sentence[i], TAGS[POS_list[i]]))
      time1 = self.output(output_filename)
      return time1
    
    def viterbi(self, sentence):
      prob = np.zeros((len(sentence), num_tags))
      prev = np.zeros((len(sentence), num_tags))

      for a in range(len(sentence)):
        sentence[a] = sentence[a].lower()

      if sentence[0] in self.words.keys():
        prob[0] = self.initial_prob_table * self.observation_prob_table[:, self.words.get(sentence[0])]
      else:
        prob[0] = self.initial_prob_table
      prev[0] = np.zeros(num_tags)
      for t in range(1,len(sentence)):
        for i in range(num_tags):
          vals =  prob[t-1,:] * self.transition_prob_table[:,i]
          x = np.argmax(vals)
          observation_ind = self.words.get(sentence[t])
          if observation_ind != None:
            prob[t,i] = prob[t-1,x] * self.transition_prob_table[x,i] * self.observation_prob_table[i,observation_ind]
          else:
            prob[t,i] = prob[t-1,x] * self.transition_prob_table[x,i]
          prev[t,i] = x
        if np.sum(prob[t,:]) == 0:
          prob[t,:] = np.sum(self.transition_prob_table, axis = 1)
          for z in range(num_tags):
            prev[t,z] = np.argmax(prob[t-1,:])
      #prob = normalize(prob, axis = 1, norm ='l1')
      prob = prob/(prob.sum(axis=1)[:,None])
      return prob, prev

    def backtrack(self, probabilities, indicies_of_previous, sentence_length):
      indicies = []
      final_POS_tag = np.argmax(probabilities[sentence_length-1])
      indicies.append(int(final_POS_tag))
      backtrack_index = sentence_length
      cur = final_POS_tag
      while len(indicies) < indicies_of_previous.shape[0]:
        backtrack_index -= 1
        prev = int(indicies_of_previous[backtrack_index, cur])
        indicies.insert(0,prev)
        cur = prev
      return indicies

    def output(self, output_filename):
        """
        Puts the new prediction in output file

        :param state: The current state of which we are printing its parents.
        :type filename: State class
        :param file: The name of the given file to print the moves.
        :type filename: str
        :return: Nothing
        :rtype: None
        """
        start_output = time.time()
        #print("Output Starting")
        output_file = open(output_filename, "w")
        for i in range(len(self.predictions)):
            (word, pos) = self.predictions[i]
            output_file.write(word) 
            output_file.write(" : ")
            output_file.write(pos)
            if i == len(self.predictions)-1:
              return start_output
            output_file.write("\n") 
        return start_output

def accuracy(predicted, answers):
  correct = 0
  total = 0
  file1 = open(predicted, "r")
  file2 = open(answers, "r")
  incorrect_list = np.zeros((91,91))
  for line1 in file1:      
    for line2 in file2:
      if line1 == line2: 
        correct +=1   
      else:
        if line2[-1] == '\n':
          incorrect_list[TAGS.index(line2[line2.index(" : ")+3:line2.index("\n")]), TAGS.index(line1[line1.index(" : ")+3:line1.index("\n")])] += 1
      total += 1 
      break
  return correct/total, incorrect_list



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainingfiles",
        action="append",
        nargs="+",
        required=True,
        help="The training files."
    )
    parser.add_argument(
        "--testfile",
        type=str,
        required=True,
        help="One test file."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The output file."
    )
    args = parser.parse_args()
    start_time = time.time()
    training_list = args.trainingfiles[0]
    # training_list = ["training2.txt"]
    #print("training files are {}".format(training_list))
    prediction_model = HMM(training_list)
    prediction_time = time.time()
    #print("Training Complete")
    #print("Starting the tagging process.")

    testfile = args.testfile
    output_file = args.outputfile
    # testfile = "test1.txt"
    # output_file = "output74.txt"
    output_start_time = prediction_model.predictor(testfile, output_file, start_time)

    # print("test file is {}".format(args.testfile))

    # print("output file is {}".format(args.outputfile))

    # correct_file = "training1.txt"
    # end_time = time.time()

    # final_accuracy, incorrect_matrix = accuracy(output_file, correct_file)
    # print("Final Accuracy: ", final_accuracy)
    # print("Training Time: ", prediction_time-start_time)
    # print("Tagging time: ", output_start_time-prediction_time)
    # print("Output Time: ", end_time - output_start_time)

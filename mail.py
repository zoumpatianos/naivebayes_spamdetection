from __future__ import division
import re
import os
import sys

FILENAME_LEGIT = re.compile(r".*\d+(legit)\d+\.txt")
FILENAME_SPAM = re.compile(r".*\d+(spmsg)\d+\.txt")
SUBJECT_REGEX = re.compile("Subject: (.*)")



class Email(object):
    """
    Describes an email.
    """
    target_value = None
    filename = ""
    
    def __init__(self, filename):
        self.words = []
        
        if(FILENAME_LEGIT.match(filename)):
            self.target_value = "NOT SPAM"
        elif(FILENAME_SPAM.match(filename)):
            self.target_value = "SPAM"
        else:
            print "Couldn't identify spam status of message: %s" % filename
            
        self.filename = filename
        for line in open(self.filename):
            if SUBJECT_REGEX.match(line):
                self._process_words(SUBJECT_REGEX.match(line).group(1))
            else:
                self._process_words(line)
                
                
    def _process_words(self,line):
        tmp_words = [word.strip() for word in line.split(" ")]
        if "" in tmp_words:
            tmp_words.remove("")
        self.words += tmp_words
    
    def __str__(self):
        return "IS_SPAM: %s\nFILENAME:%s \nDATA: %s" % (str(self.target_value), self.filename, ", ".join(self.words))
            

class EmailSet(object):
    """
    Describes a directory with emails.
    """
    directory = "."
    
    def __init__(self, directory):
        self.emails = []
        self.directory = directory
        self.emails = []
        self._load_emails()
        
    def _load_emails(self):
        for dirname, dirnames, filenames in os.walk(self.directory):
            for filename in filenames:
                if ".txt" in filename:
                    self.emails += [Email(os.path.join(dirname, filename))]
        
        
class EmailFoldSet(object):
    """
    Describes a set of email directories to be used for k-fold
    """
    def __init__(self, directory):
        self.email_sets = []
        for dirname, dirnames, filenames in os.walk(directory):
            for subdirname in dirnames:
                if not "unused" in subdirname:
                    self.email_sets += [EmailSet(os.path.join(dirname, subdirname))]
                    
    def test(self, classifier_klass, output):
        """
        Test a classifier class against the folds by using ome fold at a time for teting and the others 
        as training.
        """
        scores = []
        spam_recalls = []
        spam_precisions = []
        
        for test_set in range(len(self.email_sets)):
            classifier = classifier_klass()
            correct = 0
            false = 0
            
            total_spam = 0
            total_spam_identified = 0
            correctly_spam_identified = 0
            
            
            # test_set email set as testing and all other as training
            output.write( "\n\nFold: %d" % test_set )
            
            output.write( "\nLoading examples..." )
            
            for set_index, email_set in enumerate(self.email_sets):
                if set_index != test_set:
                    for email in email_set.emails:
                        #print email.filename
                        classifier.load_example(email.filename, email.words, email.target_value)
                        
            output.write( "\nTraining..." )
            
            classifier.train()
            
            output.write( "\nGuessing..." )
            
            for email in self.email_sets[test_set].emails:
                guess = classifier.guess(email.words)
                
                if email.target_value == guess[0]:
                    correct += 1
                else:
                    false += 1
                    
                if guess[0] == "SPAM":
                    total_spam_identified += 1
                    
                if email.target_value == "SPAM":
                    total_spam += 1
                    if email.target_value == guess[0]:
                        correctly_spam_identified += 1
                        
                    
            scores += [((correct/(correct+false))*100)]
            spam_recalls += [correctly_spam_identified / total_spam]
            if total_spam_identified > 0:
                spam_precisions += [correctly_spam_identified / total_spam_identified]
            else:
                spam_precisions += [0]
            
            output.write( "\nGuess success: %.3lf%%" % scores[-1] )
            output.write( "\nSpam recall: %lf" % spam_recalls[-1] )
            output.write( "\nSpam precision: %lf" % spam_precisions[-1] )
            
        
        output.write( "\n\nMEAN GUESS SUCCESS:\t%.3lf%%" % (sum(scores)/len(self.email_sets)) )
        output.write( "\nMEAN SPAM RECALL:\t%lf" % (sum(spam_recalls)/len(self.email_sets)) )
        output.write( "\nMEAN SPAM PRECISSION:\t%lf\n" % (sum(spam_precisions)/len(self.email_sets)) )
                
            
            


if __name__ == "__main__":
    import sys
    email_fold_set = EmailFoldSet(sys.argv[1])
    #print email_set.emails[1]
    #email = Email(sys.argv[1])
    #print email
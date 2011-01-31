from __future__ import division
from text_bayes import TextBayesClassifier
from mail import *

if __name__ == "__main__":
    import sys
    
    email_foldset = EmailFoldSet(sys.argv[1])
    email_foldset.test(TextBayesClassifier, output = sys.stdout)

    
    """    
    tb = TextBayesClassifier()
    for file in sys.argv[1:-1]:
        email = Email(file)
        tb.load_example(email.filename, email.words, email.target_value)
    tb.train()
    print tb.guess_2(Email(sys.argv[-1]).words)
    """
    
    
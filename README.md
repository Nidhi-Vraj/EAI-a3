##### adisoni-nsadhuva-svaddi-a3
a3 created for adisoni-nsadhuva-svaddi

# Assignment 3 - Probability and Statistical Learning

## Part 1 - POS tagging

### Aim
Mark every word in a sentence with its respective part of speech from a test file, given training data that has words labelled with their parts of speech tags

This is to be implemented using three Bayes Networks: simplified, HMM and complicated model

### Algorithm
#### Simple Chain Model
Need to calculate :  si= arg max P(Si=si|W)
##### Using Bayes Law   P(POS| Word) = P(Word|POS) * P(Speech) / P(Word)
We got p(W|Si) -> Emission Probability, and multiplied that by the P(Si)
For the words where we don't have any part of speech tags, we assigned it as default POS tag - "Noun"

#### Hidden Markov Chain Model
We used the Viterbi algorithm to find the maximum a posteriori labelling for the sentences
Need to calculate:  (s∗1, . . . , s∗N ) = arg max P (Si = si|W )

Emissions Probability -> p(W|Si)  (If a word is not present in bag, we fixed the emission prob of that to 0.0000000001)
Transition Probabilities calculated using a Matrix that stores the transitions between all Parts of speech
Matrix also contains a column that marks initial probability, which denotes the probabilites that a particular part of speech is the dirst column

##### Viterbi is defined as Vit(index , pos) = max { (Vit(index-1, prev_pos) * transition(pos | previous_pos) } * emission(word, current_pos)) 
Solve this model by using dynamic programming (Viterbi) with the transition and emission probabilities calculated, in order to obtain the most likely sequence.


#### Complex Monte-Carlo Markov Chain Model
We have used Gibbs Sampling in order to implement the complex mcmc.

-> First, we assign random parts of speech to each word - we have used our assignment from the result of the Naive Bayes algorithm 
-> Then we choose each word, assign it to all of the twelve parts of speech and find the respective posterior probabilites
-> Assign this part of speech for the word, and continue with the other words of the sentence with keeping the modified tags
-> after tags have been computed for all words in the sentence- this generates a sample
-> Many such samples need to be taken, so this is to be done for several iterations
-> Ignore Burnout Samples - we have taken this as 300 Samples
-> And total Sample Size is 900
After sampling, output the maximum occurred sequence for each word.
Accuracy for the model would change betten result as we are using random samples.


#### Difficulties faced:

- Using the hidden markov chain and the complex mcmc, the accuracies for the pos tags should have been relative higher than the simple bayes approach because they considered the transitions from previous states. Although we did not see that happening in our output – this could probably be because of overfitting of the training data files.
- Part of the challenge was parsing the file in under the time limit.
- It is also an opportunity to make sure that the solution can be written in a very scalable manner .
- We have implemented a multi-threaded solution to accomplish this alongside a near perfect OOPS.
- We have also seen the efficiency of this when we tested against against files with millions of rows to parse
- We have also noticed that the gibbs-sampling made a very high number of queries against the transitions table, nearly at one point taking up 40% of our run time. However, we have modified it to use as less as 6%
- Here is a sample call graph of how our various functions are called - img here
- Code made for transition matrix of this problem has been reused for part3

-------------------------------------------------------------------------------------------------------------------------------------------------------------------


## Part 2 - Ice tracking

### Aim
Given an image, our aim is to find two boundaries air-ice and ice-rock.

### Approach for part 1 – Simple

#### Finding air-ice boundary-
We have written code such that it finds maximum pixel intensity/strength for each column and it returns the row number of that particular maximum pixel value. 

#### Finding ice-rock boundary-
After we get the air-ice boundary, we are getting the ice-rock boundary by setting the margin as 10 pixels below the air-ice boundary. 

### Approach for part 2 
#### HMM
 --> Finding the horizon using search technique

We utilized A* search as our strategy since, as described on the Viterbi module slide on canvas, HMM inference can be posed as a search problem.

• Heuristic Function:

 #####    edge_strength[i][j] -= (c / 175) - (abs(i - x) / 175)

* Heuristic Function-
1. Finds the distance from top/total row size
2. Finds relative position w.r.t to previous max
3. So, calculating absolute distance gives us the accurate boundary and relatively smoother as higher the pixel higher the probability of being air-ice               boundary.


### Approach for part 3
#### Human Feedback
• The heuristic function we have used for this is:
 
 ##### edge_strength[i][j] -= (c / 175) - (abs(i - x) / 175)
   
* Heuristic Function here-
1. Finds the distance from top/total row size
2. Finds relative position w.r.t to human given points (x,y)
3.  So, calculating this absolute distance gives us the accurate boundary for air-ice and ice-rock displays human provided point with an asterisk (*) 
4.  When we pass command line argument in the terminal as –   python ./polar.py test_images/09.png 150 9 160 30
5.  Here, (150, 9) and (160, 30) are the column and row coordinates.

#### Difficulties Faced:

-> We attempted to build the Viterbi algorithm but found it to be quite difficult, despite the fact that the code for this part 2 is still annotated in comments. Given the constraints of time, we couldn't continue to implement it. So, in the modules part of canvas, we referred to the file for Viterbi algorithm. We then discovered that HMM inference can be posed as a search technique, therefore we went ahead and implemented a search strategy by defining the heuristic function to obtain correct air-ice and ice-rock boundaries.



-------------------------------------------------------------------------------------------------------------------------------------------------------------------	




## Part 3 - Reading Text

### Aim 
Recognize text in an image - To be implemented using simple and hidden markov model algorithms


### Algorithm
#### Simple Chain Model
For this we through simple bayes law:
Emission Probabilities - calculated by comparing the match and miss counts of astericks and spaces and attaching 
-> courier-train.png is used to train the data and calculates the probabilities for pixels 
After trying multiple weightage to matches and misses, we finally fixed 
-> Here, we have also assumed noise as 10% in the images

#### Hidden Markov Model
We implemented this using the Viterbi Algorithm:
Emission probabilities - calculated using the same way as the simple chain
Transition probabilities - probability of next letter given the current letter 
-> bc.train file used from the first question to check for initial and transition probabilities, here we ignored the POS tags that were given for each word

Dynamic Programming - calculates the max of the (prev stat's probability * transition probability of previous state to next state) and then multiply with the emission probabilty of the current state 


#### Difficulties faced
-> Assigning the right weightage for the miss/match in comparing the text image and the test images
-> Assigning a value of noise that needs to be considered in all images

##### References
-> Canvas Slides
-> https://www.youtube.com/watch?v=dZoHsVO4F3k&t=489s

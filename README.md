Requirement
-	Python 2.7
-	Dataset

How to use
-   /> python neuron.py -f {Dataset csv File} -w {Input Weight csv file} -o {Output Weight csv file} -n {Learning Rate} -t {Threshold} -l {Percent of fold}

Example
-   /> python neuron.py -f ionosphere_shuffle.csv -w weight.csv -o tenfold.csv -n -0.3 -t 0.03 -l 0.1

Input Weight File Format

An example of 3 attributes input, 2 hidden level nodes, and 1 output layer node

-   2,1 -> need to tell how many weight
-   0.2,0.4,-0.5,-0.4 -> first node in hidden layer level nodes 3 attributes and last one is bias
-   0.3,0.1,0.2,0.2 -> second node in hidden layer level
-   0.3,-0.2,0.1 -> first node in output layer node

"""
Main script used to run all the experiments generating plots and data for P2
"""

from flipflop import runFlipFlop
from fourpeaks import runFourPeaks
from tsp import runTSP
from knapsack import runKnapsack
from onemax import runOneMax


runFlipFlop()
runFourPeaks()
runTSP()
runOneMax()
runKnapsack()

# TODO: - Add custom parameter scaling per problem
#   - Add capability to iterate over ranges of parameters (pop size for mimic and ga, temp for sa)
#   - add NN stuff
#   - add ranges for random state
#   - figure out a good way to scale plots so everything is visible together
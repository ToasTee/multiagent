# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #We can use the heuristic from project 1 and modify it a bit

        """
        Try#1 - trying out different methods of operations to rank food priority while distancing from ghosts (need summed distance priorities)
        Win and loss conditions on successors
        if successorGameState.isWin():
            return 999999
        if successorGameState.isLose():
            return -99999999

        #Food
        foodList = newFood.asList()
        closestFood = foodList[0]
        for food in foodList:
            closestFood = min(util.manhattanDistance(newPos,food), closestFood)
        foodPriority = 100.0 / closestFood

        #Ghosts
        ghostPos = []
        if ghostState.scaredTimer == 0:
            ghostPos = [ghostState.getPosition() for ghostState in newGhostStates]
            closestGhost = min([util.manhattanDistance(newPos, ghostPosition) for ghostPosition in ghostPos])
            if closestGhost == 0:
                return -99999999
        else:
            return 999999

        scaredTime = sum(newScaredTimes)

        return successorGameState.getScore() + foodPriority - closestGhost + scaredTime"""

        """Try 2- took some hints from Q5 in using reciprocoals of values instead of flat values"""
        ghostPos = successorGameState.getGhostPositions()
        foodList = newFood.asList()

        #Food
        foodDist = 0
        for food in foodList:
            foodDist += 1.0 / manhattanDistance(newPos, food)

        #Ghost
        ghostDist = 0
        for ghost in ghostPos:
            dist = manhattanDistance(newPos, ghost)
            if dist != 0:
                ghostDist += 1.0 / dist

        return successorGameState.getScore() + foodDist - ghostDist - 100 * len(foodList)


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def minimax(gameState, currDepth, agentIndex):
            legalMoves = [action for action in gameState.getLegalActions(agentIndex)]

            # Terminal State
            if currDepth > self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            # depth updates
            nextIndex = agentIndex + 1
            nextDepth = currDepth
            if nextIndex >= gameState.getNumAgents():
                nextIndex = 0
                nextDepth += 1

            # minimax on every move
            successor = []
            for action in legalMoves:
                successor.append(minimax(gameState.generateSuccessor(agentIndex, action), nextDepth, nextIndex))

            # Max choice (pacman)
            if agentIndex == 0:
                # First Move
                bestIndex = []
                if currDepth == 1:
                    bestAction = max(successor)
                    for index in range(len(successor)):
                        if successor[index] == bestAction:
                            return legalMoves[index]
                else:
                    return max(successor)

            # Min choice (ghost)
            else:
                return min(successor)

        return minimax(gameState, 1, 0)




class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #Alpha - max best option on path to root
        #Beta - min best option on path to root
        """
        def max(min)-value(state, alpha, beta):
            initialized v = -inf
             for each successor of state:
                v = max(v, value(successor, alpha, beta))
                   if v >= beta return v (if v <= alpha return v)
                alpha = max(alpha,v) (beta = min (beta, v))
            return v
        """

        def value(state, agentIndex, depth, alpha, beta):

            def maxValue(currentState, alpha, beta):
                v = -(float('inf'))
                choice = None
                actions = currentState.getLegalActions(agentIndex)
                if len(actions) == 0:
                    return (self.evaluationFunction(currentState), None)
                for action in actions:
                    successor = currentState.generateSuccessor(agentIndex, action)
                    nextV, nextAction = value(successor, nextAgent, depth, alpha, beta)
                    if nextV > v:
                        v = nextV
                        choice = action
                    if v > beta:
                        return (v, choice)
                    alpha = max(alpha, v)
                return (v, choice)

            def minValue(currentState, alpha, beta):
                v = float('inf')
                choice = None
                actions = currentState.getLegalActions(agentIndex)
                if len(actions) == 0:
                    return (self.evaluationFunction(currentState), None)

                for action in actions:
                    successor = currentState.generateSuccessor(agentIndex, action)
                    nextV, nextAction = value(successor, nextAgent, depth, alpha, beta)

                    if nextV < v:
                        v = nextV
                        choice = action
                    if v < alpha:
                        return (v, choice)
                    beta = min(beta, v)
                return (v, choice)

            numAgent = state.getNumAgents()
            nextAgent = (agentIndex + 1) % numAgent
            if agentIndex == 0:
                depth += 1
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return (self.evaluationFunction(state), None)
            if agentIndex == 0:
                return maxValue(state, alpha, beta)
            return minValue(state, alpha, beta)

        return value(gameState, 0, -1, -(float('inf')), float('inf'))[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        """def value(state):
            if state is terminal state: return state utility
            if next is MAX: return maxValue(state)
            if next is EXP: return expValue(state)
            
            def maxValue(state):
                v= -inf
                for each successor of state:
                    v = max(v, value(successor))
                return v
            
            def expValue(state):
                v = 0
                for each successor of state:
                    p = probability(successor)
                    v += p*value(successor)
                return v
            """
        def value(state, agentIndex, depth):
            #Max
            def maxValue(currentState):
                v = -9999999
                choice = None
                actions = currentState.getLegalActions(agentIndex)
                if len(actions) == 0:
                    return self.evaluationFunction(currentState), None
                for action in actions:
                    nextState = currentState.generateSuccessor(agentIndex, action)
                    nextV, nextAction = value(nextState, nextAgent, depth)
                    if nextV > v:
                        v = nextV
                        choice = action
                return (v, choice)

            #expected value
            def expValue(currentState):
                v = 0
                choice = None
                actions = currentState.getLegalActions(agentIndex)
                if len(actions) == 0:
                    return self.evaluationFunction(currentState), None
                for action in actions:
                    nextState = currentState.generateSuccessor(agentIndex, action)
                    nextV, nextAction = value(nextState, nextAgent, depth)
                    v = v + nextV / float(len(actions))
                    choice = action
                return (v, choice)

            numAgents = state.getNumAgents()
            nextAgent = (numAgents + agentIndex + 1) % numAgents
            if agentIndex == 0:
                depth += 1
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(state), None
            if agentIndex == 0:
                return maxValue(state)
            return expValue(state)

        return value(gameState, 0, -1)[1]


        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION:
      Idea1- assign point values/weights to (in order of most important to least):
        -closest food
        -closest ghost
        -closest scared ghost
        -number of pellets/capsules left
        -number of food left
    """
    "*** YOUR CODE HERE ***"

    #Weights
    wFood = 2
    wGhost = 1
    wFoodListLen = 50

    #lists of positions
    pacmanPos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostScaredTime = [ghostState.scaredTimer for ghostState in currentGameState.getGhostStates()]
    ghostPosList = currentGameState.getGhostPositions()


    #Terminal
    #Tried changing the return "inf" to 999999 to see if that may change behavior
    if currentGameState.isWin():
        return 999999
    if currentGameState.isLose():
        return -999999

    #Ghost distances
    ghostDist = 0
    for ghost in ghostPosList:
        dist = manhattanDistance(pacmanPos, ghost)
        ghostDist += 1.0 / dist

    #Food distances
    foodDist = 0
    for food in foodList:
        foodDist += 1.0 / manhattanDistance(pacmanPos, food)

    return currentGameState.getScore() + wFood * foodDist - wGhost * ghostDist - wFoodListLen * len(foodList) + sum(ghostScaredTime)

# Abbreviation
better = betterEvaluationFunction


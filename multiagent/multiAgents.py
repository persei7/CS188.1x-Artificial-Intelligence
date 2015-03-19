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
        oldFood = currentGameState.getFood().asList()
        newFood = successorGameState.getFood().asList()
        ateCapsule = len(oldFood) - len(newFood)
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        x, y = newPos
        foodList = map(lambda f: util.manhattanDistance(newPos,f), newFood)
        #print foodList
        #print min(foodList)
        ghostFun = -3 *( 1./ (1 + min([util.manhattanDistance(newPos,g.getPosition()) \
            for g in newGhostStates])))
        capsFun = 1 * (1./(1+min(foodList+[10**100])))
        utilFun = ghostFun + ateCapsule + capsFun
        #print ghostFun, ateCapsule, capsFun, utilFun
        return utilFun
                
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
        self.numAgents = gameState.getNumAgents()
        self.numLayers = self.depth * self.numAgents
        
        res = self.value(gameState)
        #print res
        return res[1]
    
    
    def value(self, state, i=0):     
        #print i, self.numAgents, self.numLayers
        agentIndex = i % self.numAgents
        if i == self.numLayers or state.isLose() or state.isWin():
            return (self.evaluationFunction(state), None)
        
        elif agentIndex == 0:
            #pacman
            return self.maxValue(state, i)
        else:
            #ghosts
            return self.minValue(state, i)
            
    def maxValue(self, state, i):
        max_v = (-float("inf"), None)
        agentIndex = i % self.numAgents
        actions = state.getLegalActions(agentIndex)
        #print actions
        if len(actions) == 0:
            return (self.evaluationFunction(state), None)
        else:
            for a in actions:
                successor = state.generateSuccessor(agentIndex, a)
                v  = self.value(successor, i+1)
                if v[0] > max_v[0]:
                    max_v = (v[0], a)
        return max_v
        
    def minValue(self, state, i):
        min_v = (float("inf"), None)
        agentIndex = i % self.numAgents
        actions = state.getLegalActions(agentIndex)
        #print actions
        if len(actions) == 0:
            return (self.evaluationFunction(state), None)
        else:
            for a in actions:
                successor = state.generateSuccessor(agentIndex, a)
                v  = self.value(successor, i+1)
                if v[0] < min_v[0]:
                    min_v = (v[0], a)
        return min_v
        
       
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
      alpha = best already explored path for maximizer
      beta = best already explored path for minimizer
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        self.numAgents = gameState.getNumAgents()
        self.numLayers = self.depth * self.numAgents
        
        res = self.value(gameState)
        #print res
        return res[1]
    
    def value(self, state, i=0, alpha=-float('inf'), beta=float('inf')):     
        #print i, self.numAgents, self.numLayers
        agentIndex = i % self.numAgents
        if i == self.numLayers or state.isLose() or state.isWin():
            return (self.evaluationFunction(state), None)
        
        elif agentIndex == 0:
            #pacman
            return self.maxValue(state, i, alpha, beta)
        else:
            #ghosts
            return self.minValue(state, i, alpha, beta)
            
    def maxValue(self, state, i, alpha, beta):
        max_v = (-float("inf"), None)
        agentIndex = i % self.numAgents
        actions = state.getLegalActions(agentIndex)
        #print actions
        if len(actions) == 0:
            return (self.evaluationFunction(state), None)
        else:
            for a in actions:
                successor = state.generateSuccessor(agentIndex, a)
                v  = self.value(successor, i+1, alpha, beta)
                if v[0] > max_v[0]:
                    max_v = (v[0], a)
                if max_v[0] > beta:
                    return max_v
                alpha = max(alpha, max_v[0])
        return max_v
        
    def minValue(self, state, i, alpha, beta):
        min_v = (float("inf"), None)
        agentIndex = i % self.numAgents
        actions = state.getLegalActions(agentIndex)
        #print actions
        if len(actions) == 0:
            return (self.evaluationFunction(state), None)
        else:
            for a in actions:
                successor = state.generateSuccessor(agentIndex, a)
                v  = self.value(successor, i+1, alpha, beta)
                if v[0] < min_v[0]:
                    min_v = (v[0], a)
                if min_v[0] < alpha:
                    return min_v
                beta = min(beta, min_v[0])
        return min_v
    
    

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
        self.numAgents = gameState.getNumAgents()
        self.numLayers = self.depth * self.numAgents
        
        res = self.value(gameState)
        #print res
        return res[1]
    
    
    def value(self, state, i=0):     
        #print i, self.numAgents, self.numLayers
        agentIndex = i % self.numAgents
        if i == self.numLayers or state.isLose() or state.isWin():
            return (self.evaluationFunction(state), None)
        
        elif agentIndex == 0:
            #pacman
            return self.maxValue(state, i)
        else:
            #ghosts
            return self.minValue(state, i)
            
    def maxValue(self, state, i):
        max_v = (-float("inf"), None)
        agentIndex = i % self.numAgents
        actions = state.getLegalActions(agentIndex)
        #print actions
        if len(actions) == 0:
            return (self.evaluationFunction(state), None)
        else:
            for a in actions:
                successor = state.generateSuccessor(agentIndex, a)
                v  = self.value(successor, i+1)
                if v[0] > max_v[0]:
                    max_v = (v[0], a)
        return max_v
        
    def minValue(self, state, i):
        agentIndex = i % self.numAgents
        actions = state.getLegalActions(agentIndex)
        #print actions
        if len(actions) == 0:
            return (self.evaluationFunction(state), None)
        else:
            vs = []
            for a in actions:
                successor = state.generateSuccessor(agentIndex, a)                
                vs.append(self.value(successor, i+1))
        return (sum(v[0] for v in vs)/float(len(vs)), None)

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction


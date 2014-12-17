# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

import pdb

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
    "*** YOUR CODE HERE ***"
    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    #keep away from ghosts
    alertDist = 3
    ghostDists = [manhattanDistance(newPos,ghost.getPosition()) for ghost in newGhostStates]
    minGhostDist = min(ghostDists)
    if minGhostDist < alertDist:
      return float('-inf')

    foodPos = oldFood.asList()
    foodNum = successorGameState.getNumFood()
    foodDists = [manhattanDistance(newPos,coord) for coord in foodPos]
    minFoodDist = min(foodDists)
        
    return successorGameState.getScore() + 10/(minFoodDist+1) - 10*foodNum

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

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    self.agentNum = gameState.getNumAgents()
    if self.depth==0:
      return float('inf')

    actions = gameState.getLegalActions(0)
    if actions==[]:
      return self.evaluationFunction(gameState)

    vals = [self.minValue(gameState.generateSuccessor(0,move),1,self.depth) for move in actions]
    bestVal = max(vals)
    bestIndices = [index for index in range(len(vals)) if bestVal==vals[index]]
    chosenMove = random.choice(bestIndices)

    return actions[chosenMove]

  def maxValue(self,gameState,depth):
    if depth==0:
      return self.evaluationFunction(gameState)

    actions = gameState.getLegalActions(0)
    if actions==[]:
      return self.evaluationFunction(gameState)

    results = [self.minValue(gameState.generateSuccessor(0,move),1,depth) for move in actions]
    maxVal = max(results)

    return maxVal

  def minValue(self,gameState,agentIndex,depth):
    if depth==0:
      return self.evaluationFunction(gameState)

    actions = gameState.getLegalActions(agentIndex)
    if actions==[]:
      return self.evaluationFunction(gameState)

    results = []
    for move in actions:
      successor = gameState.generateSuccessor(agentIndex,move)
      if agentIndex == self.agentNum-1:
        results += [self.maxValue(successor,depth-1)]
      else:
        results += [self.minValue(successor,agentIndex+1,depth)]
    minVal = min(results)

    #pdb.set_trace()

    return minVal

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    """
      start action, in fact, it's an initial process of maxValue, 
      but it includes the firt move of pacman
    """
    alpha = float('-inf')
    beta = float('inf')

    if self.depth==0:
      return float('inf')

    actions = gameState.getLegalActions(0)
    if actions==[]:
      return self.evaluationFunction(gameState)

    bestMove = Directions.STOP
    v = float('-inf')
    for move in actions:
      curRes = self.minValue(gameState.generateSuccessor(0,move),1,self.depth,alpha,beta)
      if curRes>=v:
        v = curRes
        bestMove = move
      if v>=alpha:
        alpha = v
    return bestMove

  def maxValue(self,gameState,depth,alpha,beta):
    actions = gameState.getLegalActions(0)
    if actions==[] or depth==0:
      return self.evaluationFunction(gameState)

    maxRes = float('-inf')
    for move in actions:
      curRes = self.minValue(gameState.generateSuccessor(0,move),1,depth,alpha,beta)
      maxRes = max(curRes,maxRes)
      if maxRes>=beta:
        return maxRes
      alpha = max(alpha,maxRes)
    return maxRes


  def minValue(self,gameState,agentIndex,depth,alpha,beta):
    actions = gameState.getLegalActions(agentIndex)
    if actions==[] or depth==0:
      return self.evaluationFunction(gameState)

    minRes = float('inf')
    for move in actions:
      successor = gameState.generateSuccessor(agentIndex,move)
      if agentIndex==gameState.getNumAgents()-1: #last ghost agent
        curRes = self.maxValue(successor,depth-1,alpha,beta)
      else:
        curRes = self.minValue(successor,agentIndex+1,depth,alpha,beta)
      minRes = min(curRes,minRes)
      if alpha>=minRes:
        return minRes
      beta = min(beta,minRes)
    return minRes

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
    if self.depth==0:
      return float('inf')

    actions = gameState.getLegalActions(0)
    if actions==[]:
      return self.evaluationFunction(gameState)

    results = [self.minValue(gameState.generateSuccessor(0,move),1,self.depth) for move in actions]
    optRes = max(results)
    optIndice = [index for index in range(len(results)) if results[index]==optRes]
    chosenIndex = random.choice(optIndice)
    #pdb.set_trace()
    return actions[chosenIndex]

  def maxValue(self,gameState,depth):
    actions = gameState.getLegalActions(0)
    if actions==[] or depth==0:
      return self.evaluationFunction(gameState)

    results = [self.minValue(gameState.generateSuccessor(0,move),1,depth) for move in actions]
    #pdb.set_trace()
    return max(results)

  def minValue(self,gameState,agentIndex,depth):
    actions = gameState.getLegalActions(agentIndex)
    if actions==[] or depth==0:
      return self.evaluationFunction(gameState)

    results = []
    if agentIndex == gameState.getNumAgents()-1:
      results = [self.maxValue(gameState.generateSuccessor(agentIndex,move),depth-1) for move in actions]
    else:
      results = [self.minValue(gameState.generateSuccessor(agentIndex,move),agentIndex+1,depth) for move in actions]

    resSum = 0
    for i in results:
      resSum += i
    expRes = resSum/len(results)
    #pdb.set_trace()
    return expRes
    

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  "*** YOUR CODE HERE ***"
  pacmanPos = currentGameState.getPacmanPosition()
  foods = currentGameState.getFood()
  ghostStates = currentGameState.getGhostStates()
  #scaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

  #scaredTimerAndPositions = [(ghostState.scaredTimer, ghostState.getPosition()) for ghostState in ghostStates]
  scaredTimers = [ghostState.scaredTimer for ghostState in ghostStates]
  ghostDists = [manhattanDistance(ghostState.getPosition(),pacmanPos) for ghostState in ghostStates]

  #pdb.set_trace()

  minGhostDist = min(ghostDists)
  optIndice = [ind for ind in range(len(ghostDists)) if ghostDists[ind]==minGhostDist]
  chosenIndex = random.choice(optIndice)
  isScared = scaredTimers[chosenIndex]

  if isScared>2 and minGhostDist<10:
    scareGhost = 200+minGhostDist
  elif minGhostDist ==2:
    scareGhost = -20
  elif minGhostDist <=1:
    scareGhost = -1000
    #return float('-inf')
  else:
    scareGhost = minGhostDist


  foodPos = foods.asList()
  foodCount = currentGameState.getNumFood()
  foodDists = [manhattanDistance(curFood,pacmanPos) for curFood in foodPos]
  #pdb.set_trace()
  if foodDists != []:
    minFoodDist = min(foodDists)
  else:
    minFoodDist = float('inf')  #when there is no food any more

  val = currentGameState.getScore()+10/(minGhostDist+1)-100*foodCount+scareGhost/10
  return val

# Abbreviation
better = betterEvaluationFunction



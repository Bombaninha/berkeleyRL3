# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        '''
          Linhas da tabela: estados
          Colunas da tabela: ações
          Entrada na tabela: Valor Q de determinado estado para uma determinada ação (todos valores são inicializados com 0)
        '''

        # Podemos usar o a estrutura de Counter, presente em util.py (simula um defaultdict)
        self.Q = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """

        # Se o estado não existe na nossa tabela, acrescentamos ele
        if(state not in self.Q):
          self.Q[state] = {}

        # Se a ação não existe no nosso estado e, consequentemente não existe na tabela, acrescentamos ela, zerando o valor       
        if(action not in self.Q[state]):
          self.Q[state][action] = 0.0

        return self.Q[state][action]

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        legal_actions = self.getLegalActions(state)

        # Se não existe nenhuma ação legal, retorna None
        if(len(legal_actions) == 0):
          return None
        else:
          # Inicializa a melhor ação como sendo a primeira da lista
          first_element = list(legal_actions).pop(0)

          best_action_value = self.getQValue(state, first_element)
          best_action = first_element

          # Itera sob a primeira posição, desconsiderando a ação de posição 0
          for legal_action in legal_actions:
            # Se o valor da melhor ação for menor que a atual, realiza swap
            if(self.getQValue(state, legal_action) > best_action_value):
              best_action_value = self.getQValue(state, legal_action)
              best_action = legal_action 
            # Se o valor da melhor ação for igual ao atual, desempata. Fonte: README
            elif(self.getQValue(state, legal_action) == best_action_value):
              random_element = random.choice([legal_action, best_action])
              best_action = random_element
              best_action_value = self.getQValue(state, best_action)
            # Caso desconsiderado, não deve fazer nada
            else:
              continue

        return best_action 

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        max_action = self.computeActionFromQValues(state)

        if(max_action == None):
          return 0.0
        else:
          return self.Q[state][max_action]

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legal_actions = self.getLegalActions(state)
        action = None

        # Se existir pelo menos uma ação legal
        if(len(legal_actions) > 0):
          # Escolhemos randomicamente com probabilidade self.epsilon
          if(util.flipCoin(self.epsilon)):
            # Escolhe uma ação randômica
            action = random.choice(legal_actions)
          else:
            # Escolhe a ação de máximo valor
            action = self.getPolicy(state)
            
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """

        # Utilizando a fórmula dada no slide (46) - Aula 15
        # Criação de variáveis para deixar o mais parecido possível com a equação fornecida

        alpha = self.alpha
        r = reward
        gamma = self.discount

        # Optamos por essa abstração, uma vez que foi alertado pelo README dessa decisão
        max_value = self.getValue(nextState)

        self.Q[state][action] = ((1 - self.alpha) * self.getQValue(state, action)) + self.alpha * (r + gamma * max_value)

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass

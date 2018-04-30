
from blackjackEnvAbs import BlackjackEnvAbs
import numpy as np
from functools import reduce
import math

class BlackjackEnv(BlackjackEnvAbs):

    def _get_next_cards_prob(self):

        if self.state["nextCardIndex"] == self.cards:
            cards_in_deck = np.sum(self.state["deckCardsCount"])
            next_cards_prob = []
            for card in range(self.cards):
                specific_card_ammount = self.state["deckCardsCount"][card]
                if (specific_card_ammount != 0):
                    card_prob = specific_card_ammount/cards_in_deck
                    next_cards_prob.append((card,card_prob))
            return next_cards_prob
        else:
            return np.array([(self.state["nextCardIndex"], 1.0)])

    def get_next_state_prob(self, action):

        def take_top_card():
            next_cards_prob = self._get_next_cards_prob() # for the deck [2,1]: [(0, 2/3), (1, 1/3)]
            states_prob = []

            for card_prob in next_cards_prob:       
                card = int(card_prob[0])
                prob = card_prob[1]   
                reward = 0      
                
                new_card_value = 0
                new_card_specific_ammount = self.state["deckCardsCount"][card]
                
                if(self.state["deckCardsCount"][card] != 0):
                    new_card_value = card + 1
                    new_card_specific_ammount = new_card_specific_ammount - 1

                newState = {
                    "valueCardsInHand": self.state["valueCardsInHand"] + new_card_value,
                    "nextCardIndex": self.state["nextCardIndex"],
                    "deckCardsCount": self.state["deckCardsCount"].copy()
                }
                newState["deckCardsCount"][card] = new_card_specific_ammount
                states_prob.append(((newState, reward), prob))

            return states_prob

        def peek_top_card():
            next_cards_prob = self._get_next_cards_prob()
            states_prob = []
            
            for card_prob in next_cards_prob:       
                card = int(card_prob[0])
                prob = card_prob[1]  
                reward = self.peek_reward      
                newState = {
                    "valueCardsInHand": self.state["valueCardsInHand"],
                    "nextCardIndex": card,
                    "deckCardsCount": self.state["deckCardsCount"].copy()
                }
                states_prob.append(((newState, reward), prob))

            return states_prob

        def quit_game():
            states_prob = [(({
                "valueCardsInHand": self.state["valueCardsInHand"],
                "nextCardIndex": self.cards,
                "deckCardsCount": np.full(self.cards, 0, dtype=np.int8)
            }, self.state["valueCardsInHand"]), 1)]
            return states_prob

        return {
            0: take_top_card,
            1: peek_top_card,
            2: quit_game
        }[action]()

        
    def _step(self, action):  
        #print("Action:", parseAction(action))
        states_prob = self.get_next_state_prob(action)
        states_reward = list(map(lambda s_p: s_p[0], states_prob))
        probs = list(map(lambda s_p: s_p[1], states_prob))
        next_index = self.np_random.choice(
            len(states_reward), p=probs)
        print("Chosen card:", next_index + 1)
        return states_reward[next_index]

    def _is_end(self):
        cards_in_deck = np.sum(self.state["deckCardsCount"])
        value_in_hand = self.state["valueCardsInHand"]
        return (cards_in_deck == 0 or value_in_hand >= self.hand_limit)
        
    def get_all_states(self):        

        to_expand_search = []
        to_expand_search.append(self.flatten_state(self.state))
        all_states = []
        while len(to_expand_search) != 0:     
            to_explore = to_expand_search.pop()            
            all_states.append(to_explore) 
            print(to_explore)   
            self.set_state(self.unflatten_state(to_explore))  
            if not self._is_end():        
                for act in range(3):
                    states_prob = self.get_next_state_prob(act)           
                    for state_prob in states_prob:
                        state = state_prob[0][0]
                        if self.flatten_state(state) not in all_states:
                            to_expand_search.append(self.flatten_state(state))
        return all_states


def policy_evaluation(policy, env, discount=1):
    states = list(map(lambda s: env.flatten_state(s), env.get_all_states()))
    values = dict(map(lambda s: (s, 0), states))# dictionary of flat state to state value
    
    raise Exception('TODO: implement')

    return values


def value_optimization(env, discount=1):
    states = list(map(lambda s: env.flatten_state(s), env.get_all_states()))
    action = dict(map(lambda s: (s, 0), states))# dictionary of flat state to action to take
    raise Exception('TODO: implement')
    
    def value_optimization_policy(state, env1):
        return action[env1.flatten_state(state)]
    return value_optimization_policy

#def parseAction(key) :
 #   choices = {0: 'take_top_card', 1: 'peek_top_card', 2: 'quit_game'}
  #  return choices.get(key, 'default')    
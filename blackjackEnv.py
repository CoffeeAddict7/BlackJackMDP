from blackjackEnvAbs import BlackjackEnvAbs
import numpy as np
from functools import reduce
import math
from operator import itemgetter

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
            if (len(next_cards_prob) == 0):
                states_prob = quit_game()
            else:
                for card_prob in next_cards_prob:       
                    card = int(card_prob[0])
                    prob = card_prob[1]   
                    reward = 0

                    new_value = self.state["valueCardsInHand"] + card + 1
                    new_next = self.cards
                    new_deck_count = self.state["deckCardsCount"].copy()

                    if(new_value > self.hand_limit):
                        new_deck_count = np.full(self.cards, 0, dtype=np.int8)
                    else: 
                        new_deck_count[card] -= 1
                    
                    newState = {
                        "valueCardsInHand": new_value,
                        "nextCardIndex": new_next, 
                        "deckCardsCount": new_deck_count
                    }                    
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
        states_prob = self.get_next_state_prob(action)
        states_reward = list(map(lambda s_p: s_p[0], states_prob))
        probs = list(map(lambda s_p: s_p[1], states_prob))
        next_index = self.np_random.choice(
            len(states_reward), p=probs)
        return states_reward[next_index]

    def _is_end(self):
        cards_in_deck = np.sum(self.state["deckCardsCount"])
        value_in_hand = self.state["valueCardsInHand"]
        return cards_in_deck == 0 or value_in_hand > self.hand_limit # Check end cond 
        
    def get_all_states(self):        
        to_expand_search = [self.flatten_state(self.state)]
        all_states = [self.flatten_state(self.state)]
        while len(to_expand_search) != 0:             
            to_explore = to_expand_search.pop()              
            self.set_state(self.unflatten_state(to_explore))              
            if not self._is_end():               
                for act in range(3):
                    states_prob = self.get_next_state_prob(act)           
                    for state_prob in states_prob:
                        state = state_prob[0][0]
                        if self.flatten_state(state) not in all_states:
                            to_expand_search.append(self.flatten_state(state))
                            all_states.append(self.flatten_state(state))        
        self.reset() 
        return all_states

def policy_evaluation(policy, env, discount=1):
    values = dict(map(lambda s: (s, 0), env.get_all_states()))# dictionary of flat state to state value    
    for state in values:        
        action = get_policy_action(env, policy, state)
        env.set_state(env.unflatten_state(state))
        if not env._is_end():
            prob_states = env.get_next_state_prob(action)
            dp_solve_utility(values, state, prob_states, env, discount) 
        env.reset()
    return values

def get_policy_action(env, policy, state):
    return policy(env.unflatten_state(state), env)

def dp_solve_utility(values, prev_state, prob_states, env, discount):
    for p_s in prob_states:
        new_state = env.flatten_state(p_s[0][0])
        reward = p_s[0][1]
        prob = p_s[1]
        values[prev_state] += prob*(reward + discount*values[new_state])
 

def value_optimization(env, N=1, discount=1):
    all_states = env.get_all_states()
    strategy_opt = dict(map(lambda s: (s, 0), all_states))
    v_opt = dict(map(lambda s: (s, 0), all_states))    
    for i in range(N):
        v_step = dict(map(lambda s: (s, 0), all_states))
        for s in all_states:
            strategy_opt[s], v_step[s] = max_action_Q(strategy_opt[s], env, v_opt, s, discount)
        v_opt = v_step
    
    def value_optimization_policy(state, env1):
        return strategy_opt[env1.flatten_state(state)]
    return value_optimization_policy

def max_action_Q(pa, env, v_opt, state, discount):
    a_value_maximizer = dict()
    env.set_state(env.unflatten_state(state))
    optimal_act = pa
    step_val = v_opt[state]

    if not env._is_end():   
        for a in range(3):
            a_value_maximizer[a] = sum(map(lambda p_s: state_utility(v_opt, state, p_s, env, discount), env.get_next_state_prob(a)))        
        optimal_act =  max(a_value_maximizer.items(), key=itemgetter(1))[0]
        step_val =  max(a_value_maximizer.items(), key=itemgetter(1))[1]
    env.reset()
    return optimal_act, step_val   

def state_utility(values, state, p_s, env, discount):     
    new_state = env.flatten_state(p_s[0][0])
    reward = p_s[0][1]
    prob = p_s[1]
    return prob * (reward + discount*(values[new_state]))
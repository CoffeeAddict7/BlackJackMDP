
from blackjackEnvAbs import BlackjackEnvAbs
import numpy as np
from functools import reduce
import math

class BlackjackEnv(BlackjackEnvAbs):

    def _get_next_cards_prob(self):

        if self.state["nextCardIndex"] == self.cards:
            cards_in_deck = np.sum(self.state["deckCardsCount"])
            raise Exception('TODO: implement')

            return card_prob
        else:
            return np.array([(self.state["nextCardIndex"], 1.0)])

    def get_next_state_prob(self, action):

        def take_top_card():
            next_cards_prob = self._get_next_cards_prob() # for the deck [2,1]: [(0, 2/3), (1, 1/3)]

            states_prob = []

            raise Exception('TODO: implement')

            return states_prob

        def peek_top_card():
            next_cards_prob = self._get_next_cards_prob()
            states_prob = []

            raise Exception('TODO: implement')

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
        raise Exception('TODO: implement')

    def get_all_states(self):

        raise Exception('TODO: implement')

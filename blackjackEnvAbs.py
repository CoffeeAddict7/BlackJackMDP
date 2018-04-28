import gym
from gym import error, spaces, utils
from gym.utils import seeding
from functools import reduce
import copy
import numpy as np


class BlackjackEnvAbs(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, cards=12, multiplicity=4, hand_limit=10, peek_reward=-1, allow_set_state=True):
        self.cards = cards
        self.multiplicity = multiplicity
        self.allow_set_state = allow_set_state
        self.hand_limit = hand_limit
        self.peek_reward = peek_reward
        self.np_random = np.random.RandomState(seed=None)

        self.observation_space = spaces.Dict({
            # 0..(sum all cards)*multiplicity
            "valueCardsInHand":  spaces.Discrete(int((self.cards+1)*self.cards/2*self.multiplicity)+1),
            # 0..cards
            # if the next card in unknown, the value is cards
            "nextCardIndex": spaces.Discrete(self.cards + 1),
            # [] to [0_1...0_cards]
            # Number of cards of each value in the deck
            # of the form [x_1, x_2, ... x_cards] where x_i is the number of cards in the deck with value i
            # example:
            # cards=3, multiplicity=2
            # deckCardsCount: [2, 1, 0]
            # deck content:
            #   - 2 cards value 1
            #   - 1 card  value 2
            #   - 0 cards value 3
            "deckCardsCount": spaces.MultiDiscrete(list(map(lambda x: self.multiplicity+1, range(self.cards))))
        })

        # 0: Takes the card from the top of the deck
        # 1: Peeks the top card
        # 2: Quits the game
        # if the players peeks twice in a row, the action causes no change in the state
        self.action_space = spaces.Discrete(3)

        self.state = {
            "valueCardsInHand": 0,
            "nextCardIndex": self.cards,
            "deckCardsCount": np.full(self.cards, self.multiplicity, dtype=np.int8)
        }

    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed=None)
        return [seed]

    def step(self, action):
        if (not(self._check_action(action))):
            return self.state, 0, self._is_end(), {}

        self.state = copy.deepcopy(self.state)

        next_state_reward = self._step(action)
        self._check_state(next_state_reward[0])
        self.state, reward = next_state_reward
        return self.state, reward, self._is_end(), {}

    def _check_action(self, action):
        if (not(self.action_space.contains(action))):
            raise Exception('Action not in action_space')

        if (self.state["nextCardIndex"] != self.cards and action == 1):
            return False

        return True

    def _check_state(self, state):
        if (not(self.observation_space.contains(state) and (0 <= self.state["deckCardsCount"]).all())):
            raise Exception('State not in observation_space')

    def get_next_state_prob(self, action):
        raise Exception('TODO: extend class and implement')

    def step_possible_results(self, action):
        raise Exception('TODO: extend class and implement')

    def _step(self, action):
        raise Exception('TODO: extend class and implement')

    def _is_end(self):
        raise Exception('TODO: extend class and implement')

    def get_all_states(self):
        raise Exception('TODO: extend class and implement')

    def reset(self):
        self.state = {
            "valueCardsInHand": 0,
            "nextCardIndex": self.cards,
            "deckCardsCount": np.full(self.cards, self.multiplicity, dtype=np.int8)
        }
        return self.state

    def set_state(self, newState):
        if (not(self.allow_set_state)):
            raise Exception('The env is configured to reject set_state')

        self._check_state(newState)
        self.state = newState.copy()

    def render(self, mode='human', close=False):
        if close:
            return

        def j6(n): return str(n).rjust(6)

        def j2(n): return str(n).rjust(2)

        titles = " |".join([
            j6("value"),
            j6("card"),
            ",".join(
                map(lambda i_n: j2(i_n[0][0]+1), np.ndenumerate(self.state["deckCardsCount"])))
        ])

        print(titles)
        # print('-' * len(titles))

        print(" |".join([
            j6(self.state["valueCardsInHand"]),
            j6("-"
                if self.state["nextCardIndex"] == self.cards else (self.state["nextCardIndex"] + 1)),
            ",".join(map(j2, self.state["deckCardsCount"]))
        ]))
        print()

    def flatten_state(self, s):
        return (s["valueCardsInHand"], s["nextCardIndex"], tuple(s["deckCardsCount"]))

    def unflatten_state(self, s):
        return {
            "valueCardsInHand": s[0],
            "nextCardIndex": s[1],
            "deckCardsCount": np.array(s[2])
        }

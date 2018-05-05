from blackjackEnv import BlackjackEnv, policy_evaluation, value_optimization
import random
from operator import itemgetter

def policy_with_caution(state, env):
    hand_limit = env.hand_limit
    max_card = env.cards
    valueCardsInHand = state["valueCardsInHand"]
    nextCardIndex = state["nextCardIndex"]
    deckCardsCount = state["deckCardsCount"]
    if (nextCardIndex != max_card):  # there is a peeked card
        if (valueCardsInHand + nextCardIndex+1) <= hand_limit:
            # the peeked card + valueCardsInHand is less or equal than the limit
            action = 0
        else:
            # the peeked card + valueCardsInHand is bigger than the limit, better quit
            action = 2
    elif valueCardsInHand + max_card < hand_limit:
        # if the bigger card of the game is drawn, it is still under the limit so there is no need to check
        action = 0
    else:
        # the next card may cause valueCardsInHand to be over the limit, better check
        action = 1

    return action


def policy_peek_before_take(state, env):
    hand_limit = env.hand_limit
    max_card = env.cards
    valueCardsInHand = state["valueCardsInHand"]
    nextCardIndex = state["nextCardIndex"]
    if nextCardIndex != max_card:
        if(valueCardsInHand + nextCardIndex + 1) <= hand_limit:
            action = 0
        else:
            action = 2
    else:
        action = 1

    return action

def policy_always_take(state, env):
    return 0


def policy_random(state, env):
    return random.choice([0, 1, 2])


def policy_runner(strategy, env, render=False):
    done = False
    state = env.reset()
    reward = 0
    while (not(done)):
        if (render):
            env.render()
        action = strategy(state, env)
        state, new_reward, done, _ = env.step(action)
        reward += new_reward    
    return reward

def main():
    '''    
    def takeCards(elem):
        return elem[2]
    
    all_states = BlackjackEnv().get_all_states()
    sorted_states = sorted(all_states,key = takeCards)
    for s in sorted_states: print(s)
    print(max(list(map(lambda i: i[0], all_states))))
    '''

    for policy in [
        # add your policies here
       # policy_random,
         policy_with_caution,
         policy_peek_before_take,       
         value_optimization(BlackjackEnv(),3),
       # policy_always_take,        
    ]:
        
        if (policy_random != policy):
            env = BlackjackEnv()
            values = policy_evaluation(policy, env)

            print("States with value: ", len(values),
                  "start state value: ", values[env.flatten_state(env.reset())])
            total = 0
            for k in values: total+=values[k]
            print("Total value", total)
        
        reward_sum = 0
        for i in range(10):
            env = BlackjackEnv()
            env.seed(i)
            reward = policy_runner(policy, env)
            print(reward, policy.__name__)
            reward_sum += reward
        print("Total", reward_sum, policy.__name__)
        print()

if __name__ == "__main__":
    # execute only if run as a script
    main()

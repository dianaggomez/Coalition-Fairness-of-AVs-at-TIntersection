from world import World
from coalition import Coalition
import numpy as np
import time

#hyperparameters
GAMMA = 0.95
ALPHA = 0.1

#Create the k coalitions
S1 = Coalition(coalition_num=1,num_of_vehicles=6,policy ='greedy')
S2 = Coalition(coalition_num=2,num_of_vehicles=6,policy ='greedy')

coalitions = [S1,S2]
world = World(coalitions)
world.fairness = False
# world.initialize_state_space()

max_episode = 50000000 #200000000
episode = 0
t = 0
number =0
# side = 1

for s in coalitions:
    if s.policy == 'greedy':
        #Initialize Q
        s.Q = np.zeros((world.state_space_size, 4))

# left side goes first, grab coalition number
# k = world.current_coalition

# Initialize s
# episode_return = []
s_t = world.current_state_idx
# Both agents will choose an action
# Choose a from s using policy dervides from Q (e-greedy)
a_t1 = S1.choose_action(s_t)
a_t2 = S2.choose_action(s_t)

R_1 = []
R_2 = []

R_eps_avg_1 = [] # return
R_eps_avg_2 = []

R_eps_1 = []
R_eps_2 = []

# world.visualize()
while t <= max_episode:
    k = world.current_coalition
    # print('coalition S : ', k)
    # print('current side: ', world.current_side)
    # print('action: ', a_t)
    # take action a, observe r, s'
    # print('actions: ', a_t1, a_t2)
    r_t, s_tp1 = world.observe((a_t1,a_t2))
    # print('reward:', r_t)
    # R_t += r_t
    # choose a' from s' using policy derived from Q (e-greedy)
    a_tp1_1 = S1.choose_action(s_t)
    a_tp1_2 = S2.choose_action(s_t)

    # world.visualize()
    # time.sleep(0.5)
    # Update both Q-tables
    S1.Q[s_t, a_t1] = S1.Q[s_t, a_t1] + ALPHA*(r_t[0] + GAMMA*S1.Q[s_tp1, a_tp1_1] - S1.Q[s_t, a_t1])
    S2.Q[s_t, a_t2] = S2.Q[s_t, a_t2] + ALPHA*(r_t[1] + GAMMA*S2.Q[s_tp1, a_tp1_2] - S2.Q[s_t, a_t2])
    # print(S1.Q[s_t, a_t])

    R_eps_1.append(r_t[0])
    R_eps_2.append(r_t[1])

    if world.is_terminal():
        # episode_return.append([R_t,world.current_timestep])
        episode +=1
        # print('Episode:', episode)
        # print('Time to clear queue: ', world.current_timestep)
        # break
        # print('return: ',R_t)
        # print('t_s1: ', S1.t_pi)
        # print('t_s2: ', S2.t_pi)
        # R_t = 0

        # R_eps_avg_1.append(np.mean(np.array(R_eps_1)))
        # R_eps_avg_2.append(np.mean(np.array(R_eps_2)))

        R_eps_avg_1.append(np.sum(np.array(R_eps_1)))
        R_eps_avg_2.append(np.sum(np.array(R_eps_2)))

        # print(R_eps_avg_1)
        # print(R_eps_avg_2)
        if episode%200 == 0:
            R_1.append(np.mean(np.array(R_eps_avg_1)))
            R_2.append(np.mean(np.array(R_eps_avg_2)))

        # Initialize the rewards
        R_eps_avg_1 = [] 
        R_eps_avg_2 = []

        R_eps_1 = []
        R_eps_2 = []

        #reset the world
        world.reset(random=True, train=True)
        s_t = world.current_state_idx

        a_t1 = S1.choose_action(s_t)
        a_t2 = S2.choose_action(s_t)
        # world.visualize()
    else:
        s_t = s_tp1 
        a_t1 = a_tp1_1
        a_t2 = a_tp1_2
    t += 1


    ALPHA = ALPHA - 0.1/max_episode
    if t%1000==0:
        print('t: ',t)
    if t%50000000==0 and t<max_episode:
        if world.fairness:
            np.save('Q_{ep}_S1_fairness'.format(ep = episode), S1.Q)
            np.save('Q_{ep}_S2_fairness'.format(ep = episode), S2.Q)
            R_t = np.stack((np.array(R_1), np.array(R_2)))
            np.save('Rt_{ep}_episodes_S1_fairness'.format(ep = episode), R_1)
            np.save('Rt_{ep}_episodes_S2_fairness'.format(ep = episode), R_2)
            number +=1
        else:
            np.save('Q_{ep}_S1'.format(ep = episode), S1.Q)
            np.save('Q_{ep}_S2'.format(ep = episode), S2.Q)
            R_t = np.stack((np.array(R_1), np.array(R_2)))
            np.save('Rt_{ep}_episodes_S1'.format(ep = episode), R_1)
            np.save('Rt_{ep}_episodes_S2'.format(ep = episode), R_2)
            number +=1


if world.fairnes:
    np.save('Q_{ep}_episodes_S1_fairness'.format(ep = episode), S1.Q)
    np.save('Q_{ep}_episodes_S2_fairness'.format(ep = episode), S2.Q)

    R_t = np.stack((np.array(R_1), np.array(R_2)))
    np.save('Rt_{ep}_episodes_S1_fairness'.format(ep = episode), R_1)
    np.save('Rt_{ep}_episodes_S2_fairness'.format(ep = episode), R_2)
    print('Q table saved')
else:
    np.save('Q_{ep}_episodes_S1'.format(ep = episode), S1.Q)
    np.save('Q_{ep}_episodes_S2'.format(ep = episode), S2.Q)

    R_t = np.stack((np.array(R_1), np.array(R_2)))
    np.save('Rt_{ep}_episodes_S1'.format(ep = episode), R_1)
    np.save('Rt_{ep}_episodes_S2'.format(ep = episode), R_2)
    print('Q table saved')

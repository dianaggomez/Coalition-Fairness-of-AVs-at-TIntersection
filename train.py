from world import World
from coalition import Coalition
import numpy as np
import time

#hyperparameters
GAMMA = 0.95
ALPHA = 0.1

#Create the k coalitions
S1 = Coalition(coalition_num=1,num_of_vehicles=6,policy ='greedy')
S2 = Coalition(coalition_num=2,num_of_vehicles=6,policy ='SR')

coalitions = [S1,S2]
world = World(coalitions)
world.fairness = False
# world.initialize_state_space()

max_episode = 10000000
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
episode_return = []
s_t = world.current_state_idx
# Choose a from s using policy dervides from Q (e-greedy)
a_t = S1.choose_action(s_t)

R_t_50 = []
R_t_100 = []
R_t_150 = []
R_t_200 = []

R_eps_avg_50 = 0 # return
R_eps_avg_100 = 0 # return
R_eps_avg_150 = 0 # return
R_eps_avg_200 = 0 # return

R_eps = 0


# world.visualize()
while t <= max_episode:
    # k = world.current_coalition
    # print('coalition S : ', k)
    # print('current side: ', world.current_side)
    # print('action: ', a_t)
    # take action a, observe r, s'
    r_t, s_tp1 = world.observe((a_t,3))
    # print('reward:', r_t)
    R_eps+=r_t
    
    # choose a' from s' using policy derived from Q (e-greedy)
    a_tp1 = S1.choose_action(s_t)

    # world.visualize()
    # time.sleep(0.5)
    S1.Q[s_t, a_t] = S1.Q[s_t, a_t] + ALPHA*(r_t + GAMMA*S1.Q[s_tp1, a_tp1] - S1.Q[s_t, a_t])
    # print(S1.Q[s_t, a_t])

    if world.is_terminal():
        # episode_return.append([R_t,world.current_timestep])
        episode +=1
        # print('Episode:', episode)
        # print('Time to clear queue: ', world.current_timestep)
        # print('return: ',R_t)

        # R_eps_avg += R_eps/S1.t_pi
        R_eps_avg_50 += R_eps
        R_eps_avg_100 += R_eps
        R_eps_avg_150 += R_eps
        R_eps_avg_200 += R_eps
        
        # Tracking the average reward over N episodes
        if episode%50 == 0:
            R_t_50.append(R_eps_avg_50/50)
            R_eps_avg_50 = 0
        if episode%100 == 0:
            R_t_100.append(R_eps_avg_100/100)
            R_eps_avg_100 = 0
        if episode%150 == 0:
            R_t_150.append(R_eps_avg_150/50)
            R_eps_avg_50 = 0
        if episode%200 == 0:
            R_t_200.append(R_eps_avg_200/100)
            R_eps_avg_100 = 0

        world.reset(random=True, train=True)
        s_t = world.current_state_idx
        a_t = S1.choose_action(s_t)

        # reset the rewards for the next episode
        R_eps = 0
        
        # world.visualize()
    else:
        s_t = s_tp1 
        a_t = a_tp1
    t += 1

    ALPHA = ALPHA - 0.1/max_episode
    if t%1000==0:
        print('t: ',t)
    if t%50000000==0:
        np.save('Q_{num}'.format(num=number), S1.Q)
        if world.fairness:
            np.save('Return_per_Episode_fairness_{ep}_SingleAgent'.format(ep=episode),R_t)
        else:
            np.save('Return_per_Episode_{ep}_SingleAgent_50'.format(ep=episode),R_t_50)
            np.save('Return_per_Episode_{ep}_SingleAgent_100'.format(ep=episode),R_t_100)
            np.save('Return_per_Episode_{ep}_SingleAgent_150'.format(ep=episode),R_t_150)
            np.save('Return_per_Episode_{ep}_SingleAgent_200'.format(ep=episode),R_t_200)
        number +=1


if world.fairness:
    np.save('Q_{ep}_episodes_fairness_3and2_SingleAgent'.format(ep = episode), S1.Q)
    print('Q table saved')
    np.save('Return_per_Episode_fairness_{ep}_SingleAgent'.format(ep=episode),R_t)
else:
    np.save('Q_{ep}_episodes_3and2_SingleAgent'.format(ep = episode), S1.Q)
    print('Q table saved')
    np.save('Return_per_Episode_{ep}_SingleAgent_50'.format(ep=episode),R_t_50)
    np.save('Return_per_Episode_{ep}_SingleAgent_100'.format(ep=episode),R_t_100)
    np.save('Return_per_Episode_{ep}_SingleAgent_150'.format(ep=episode),R_t_150)
    np.save('Return_per_Episode_{ep}_SingleAgent_200'.format(ep=episode),R_t_200)
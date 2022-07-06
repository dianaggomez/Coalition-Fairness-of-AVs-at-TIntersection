# Find optimal path for S1
from world import World
from coalition import Coalition
import numpy as np
import time
import pandas as pd

def compute_baseline(init_l_q,init_r_q):
	baseline = []
	time_SR = [0,0]
	for i in range(len(init_l_q)):
		baseline.append(init_l_q[i])
		baseline.append(init_r_q[i])

	baseline.reverse()
	b_line= np.array(baseline)
	# print(baseline)

	coalition_2 = b_line[np.argmax(b_line)]
	coalition_1 = b_line[np.argmin(b_line)]

	time_SR[coalition_2 - 1] = 12 - np.argmax(b_line) 
	time_SR[coalition_1 - 1] = 12 - np.argmin(b_line)
	return time_SR

# Q_table = np.load("/Users/diana/Desktop/moon_shot/3and2/Q_15567760_episodes_3and2.npy")
# Q_table = np.load("/Users/diana/Desktop/moon_shot/3and2/Q_0.npy")
# Q_table = np.load('Q_15680383_episodes_3and2.npy')
Q_table = np.load("/Users/diana/Desktop/moon_shot/3and2/SingleAgent/w_baseline/Q_15567760_episodes_3and2.npy")
for i in range(0,13):
	#Create the k coalitions
	S1 = Coalition(coalition_num=1,num_of_vehicles=i,policy ='greedy')
	S2 = Coalition(coalition_num=2,num_of_vehicles=12-i,policy ='SR')

	coalitions = [S1,S2]
	world = World(coalitions)

	queues_data = {}
	baseline_data = {}

	for j in range(3500):
		print(j)
		t = 0
		# Compute t_SR 
		t_SR = compute_baseline(world.original_l_q,world.original_r_q)
		# print(t_SR)
		# Store queues and their time steps
		q = list(reversed(world.original_l_q)) + world.original_r_q
		q_str = [str(x) for x in q]
		q_str = "".join(q_str)

		world.visualize()

		while not world.is_terminal():
			# Find max action for this Q out of all the actions
			s_t = world.current_state_idx
			a_tp1 = np.argmax(Q_table[s_t,:])

			# path.append(a_tp1)

			# take action and update world
			# 3 in second because S1 should always go or (1,1)
			# print("action: ", a_tp1)
			world.observe([a_tp1,3])
			# world.visualize()
			# time.sleep(0.5)

		# GRAB TIME - use world attributes
		queues_data[q_str] = [S1.t_pi,S2.t_pi, max(S1.t_pi,S2.t_pi), t_SR[0], t_SR[1]]
		# baseline_data[q_str] = t_SR
		world.reset(random=True,train=False)

		# print("t_S1 = ", t)
		# print("t_group = ", t2)

		# print('')
		# print('___________ NEW QUEUE__________')
		# print('')

		print("Queues: ", queues_data)
		queue_df = pd.DataFrame.from_dict(queues_data, orient='index', columns=['t_s1', 't_s2','t_group', 't_SR_S1', 't_SR_S2'])
		print(queue_df)
		# np.save('queues_{num}'.format(num=i), queues_data)
	np.save('queues_{num}'.format(num=i), queues_data)
	# np.save('baseline_{num}'.format(num=i), baseline_data)
	print('Queue', i, 'saved')


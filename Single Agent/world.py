from collections import deque
import numpy as np
from itertools import product
import time
import random

class World():
    def __init__(self, coalitions):

        self.coalitions = coalitions
        self.original_l_q, self.original_r_q = self.gen_queue()

        self.l_q = deque(self.original_l_q)
        self.r_q = deque(self.original_r_q)

        self.state_space_size = 0
        self.state_to_index = {}
        self.index_to_state = {}
        self.initialize_state_space()

        self.index_to_action = {0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)}

        self.current_state_idx = self.state_to_index[tuple(self.original_l_q + self.original_r_q)]

        # self.current_side = 'left'
        self.current_coalition = self.l_q[0]

        self.coalition_cleared = np.array([False, False])
        self.assigned = [0,0]

        self.current_timestep = 0

        self.fairness = False

    def gen_queue(self):
        queue = []
        for s in self.coalitions:
            queue += [s.coalition_num]*s.num_of_vehicles

        #shuffle queue
        queue_copy = queue.copy()
        random.shuffle(queue_copy)

        #split queue
        l_q= queue_copy[:len(queue_copy)//2]
        r_q = queue_copy[len(queue_copy)//2:]

        return l_q,r_q

    def initialize_state_space(self):
        n = len(self.coalitions)
        state_space = list(product(np.arange(0,n+1),repeat = len(self.original_l_q) + len(self.original_r_q)))
        self.state_space_size = len(state_space)
        for i, state in enumerate(state_space):
            self.state_to_index[state] = i
            self.index_to_state[i] = state

    def get_effective_action(self, action):
        ego_action = self.index_to_action[action[0]]
        op_action = self.index_to_action[action[1]]
        effective_action = []
        if len(self.l_q) > 0 and len(self.r_q) > 0:
            if self.l_q[0]==1 and self.r_q[0]==1:
                effective_action = ego_action
            elif self.l_q[0]==1 and self.r_q[0]==2:
                effective_action = [ego_action[0], op_action[1]]
            elif self.l_q[0]==2 and self.r_q[0]==1:
                effective_action = [op_action[0], ego_action[1]]
            elif self.l_q[0]==2 and self.r_q[0]==2:   
                effective_action = op_action
            side_empty = 'none'
        elif len(self.l_q) == 0:
            effective_action = (0, 1)
            side_empty = 'left'
        elif len(self.r_q) == 0:
            effective_action = (1, 0)
            side_empty = 'right'

        return effective_action, side_empty

    def take_step(self, effective_action):
        popped = 0
        last_popped = [0,0]
        vehicles_exited = []
        if effective_action[0]==0 and effective_action[1]==0:
            pass
        elif effective_action[0]==0 and effective_action[1]==1:
            if len(self.r_q) > 0:
                if self.r_q[0] == 2:
                    vehicles_exited.append(self.r_q.popleft())
                    popped+=1
                else:
                    # check that the next 3 queued are the greedy coalition
                    for i in range(min(len(self.r_q),3)):
                        if self.r_q[0] == 2:
                            break
                        else:
                            vehicles_exited.append(self.r_q.popleft())
                            popped+=1
        elif effective_action[0]==1 and effective_action[1]==0:
            if len(self.l_q) > 0:
                if self.l_q[0] == 2:
                    vehicles_exited.append(self.l_q.popleft())
                    popped+=1
                else:
                    for i in range(min(len(self.l_q),3)):
                        # check that the next 3 queued are the greedy coalition
                        if self.l_q[0] == 2:
                            break
                        else:
                            vehicles_exited.append(self.l_q.popleft())
                            popped+=1
        elif effective_action[0]==1 and effective_action[1]==1:
            if len(self.l_q) > 0:
                vehicles_exited.append(self.l_q.popleft())
                last_popped[0] = vehicles_exited[-1]
                popped+=1

            if len(self.r_q) > 0:
                vehicles_exited.append(self.r_q.popleft())
                last_popped[1] = vehicles_exited[-1]
                popped+=1
        return popped, last_popped, vehicles_exited

    def is_end(self,coalition_num):
        # need to check for both vehilces
        # Currently only checking for coalition 2 - then assign -2 reward
        if  coalition_num not in self.l_q and coalition_num not in self.r_q:
            self.coalition_cleared[coalition_num-1] = True
            self.assigned[coalition_num-1] +=1
            self.coalitions[coalition_num-1].t_pi = self.current_timestep
            return True
        else:
            return False

    def observe(self, action):
        # take action

        # Need to check the condition for each:
        # (0,0): no one goes
        # (0,1): 3 vehicles go
        # (1,0): 3 vehicles go
        # (1,1): 2 vehicles go

        # get the effective action
        effective_action, side_empty = self.get_effective_action(action)
        # print('Action:',effective_action)
        popped, last_popped, vehicles_exited = self.take_step(effective_action)


        # self.current_timestep += 1
        for i in range(1,3):
            self.is_end(i)
        # print(self.coalition_cleared)
        # print(self.assigned)
        # gather reward

        # print(self.assigned, self.coalition_cleared)

        # gather state
        state_l_q = list(self.l_q) + [0]*(len(self.original_l_q) - len(self.l_q))
        state_r_q = list(self.r_q) + [0]*(len(self.original_r_q) - len(self.r_q))

        state = self.state_to_index[tuple(state_l_q + state_r_q)]
        self.current_state_idx = state

        
        if side_empty == 'none' and 1 not in self.index_to_state[self.current_state_idx] and 2 not in self.index_to_state[self.current_state_idx]:
            if last_popped == [2,1]:
                reward = -2
            elif last_popped == [1,2] or last_popped == [1,1]:
                reward = 1
         
        elif self.assigned[1] == 1 and self.coalition_cleared[1]:
            reward = -2
            self.assigned[1] +=1
        elif self.assigned[0] == 1 and self.coalition_cleared[0]:
            reward = 1
            self.assigned[1] +=1
        else:
            reward = -1

        # self.update_turn()
        if side_empty == 'none':
            self.current_timestep += 2
        else:
            if popped == 1:
                self.current_timestep += 1
            else:
                self.current_timestep += 2

        # Calculate fairness reward
        if self.fairness: 
            # vehicles_exit: are the vehicles that exited on each turn
            if (np.array(vehicles_exited) == 1).all() and len(vehicles_exited)>0:
                r_f = (self.coalitions[1].num_of_vehicles/12)/len(vehicles_exited)
            else:
                # r_f = 0
                r_f = (self.coalitions[1].num_of_vehicles/12)/(2/3)

            reward = reward - r_f
        # print("S2 vehicles: ", self.coalitions[1].num_of_vehicles)
        # print("Vehicles exited: ", vehicles_exited)
        # print('fairness reward: ', r_f)
        # print("reward: ", reward)
        return reward, state



    def is_terminal(self):
        state = self.index_to_state[self.current_state_idx]
        if 1 not in state:
            self.coalitions[0].t_pi = self.current_timestep
            if self.coalitions[1].t_pi == 0:
                # print(sum(np.asarray(state)==2))
                self.coalitions[1].t_pi = self.current_timestep + sum(np.asarray(state)==2)
            return True
        else:
            return False

    def reset(self, random=False, train=True):
        if random:
            if train:
                init_l_q, init_r_q = self._generate_queue_randomly(len(self.original_l_q), len(self.original_r_q))
                self.l_q = deque(init_l_q)
                self.r_q = deque(init_r_q)
                self.original_l_q = init_l_q
                self.original_r_q = init_r_q
            else:
                # init_l_q, init_r_q = self._generate_queue_randomly(train=False, l_q=self.original_l_q, r_q=self.original_r_q)
            # self.l_q = deque(init_l_q)
            # self.r_q = deque(init_r_q)
            # self.original_l_q = init_l_q
            # self.original_r_q = init_r_q
                self.original_l_q, self.original_r_q = self.gen_queue()

                self.l_q = deque(self.original_l_q)
                self.r_q = deque(self.original_r_q)

        else:
            self.l_q = deque(self.original_l_q)
            self.r_q = deque(self.original_r_q)

        #rest attributes
        self.current_state_idx = self.state_to_index[tuple(self.original_l_q + self.original_r_q)]
        # self.current_side = 'left'
        # self.current_coalition = self.l_q[0]

        self.coalition_cleared = np.array([False, False])
        self.assigned = [0,0]

        self.current_timestep = 0
        self.coalitions[0].t_pi = 0
        self.coalitions[1].t_pi = 0
    def _generate_queue_randomly(self, l_q_len=0, r_q_len=0, train=True):

        if train:
            l_q = np.random.choice([1, 2], len(self.original_l_q), p=[0.5, 0.5])
            r_q = np.random.choice([1, 2], len(self.original_r_q), p=[0.5, 0.5])

            while list(r_q).count(1)==0 ^ list(l_q).count(1)==0:
                l_q = np.random.choice([1, 2], len(self.original_l_q), p=[0.5, 0.5])
                r_q = np.random.choice([1, 2], len(self.original_r_q), p=[0.5, 0.5])

            return list(l_q), list(r_q)
        else:
            l_q = self.original_l_q.copy()
            random.shuffle(l_q)
            r_q = self.original_r_q.copy()
            random.shuffle(r_q)

            return l_q, r_q

    def visualize(self):
        state = self.index_to_state[self.current_state_idx]
        state_l_q = state[:len(self.original_l_q)]
        state_r_q = state[len(self.original_r_q):]
        string = '|'
        for cell in reversed(state_l_q):
            string += str(cell) + '|'
        string += '*|'
        for cell in state_r_q:
            string += str(cell) + '|'
        print(string)



# if __name__ == '__main__':

#     GAMMA = 0.95
#     LAMBDA = 0.9
#     ALPHA = 0.1

#     init_l_q = [2, 2, 1, 1, 1, 1]
#     init_r_q = [1, 1, 2, 2, 2, 2]
#     world = World(init_l_q, init_r_q)
    
#     t = 0
#     s_t = world.observe()
#     a_t = 3

#     num_state = 3**(len(init_l_q) + len(init_r_q))
#     Q = np.zeros((num_state, 4))
#     N = np.zeros((num_state, 4))

#     Q_last = Q.copy()
#     try:
#         while True:
#             r = world.step([a_t, 3])
#             # S2 always chooses to go 3 -> (1,1)
#             s_tp1 = world.observe()
#             a_tp1 = choose_action(s_tp1, Q)
#             N[s_t, a_t] = N[s_t, a_t] + 1
#             delta = r + GAMMA*Q[s_tp1, a_tp1] - Q[s_t, a_t]

#             Q = Q + ALPHA*delta*N
#             N = GAMMA*LAMBDA*N

#             if world.is_end():
#                 world.reset()
#                 N = np.zeros((num_state, 4))
#                 s_t = world.observe()
#                 a_t = 3

#             d = np.linalg.norm(Q - Q_last)
#             # print('Iter: {:d}    D:{:.5f}    r:{:.2f}'.format(t, d, r))
#             # if d <= 0.5:
#                 # break
#             # if d <= 0.0001 and t > 1000:
#             #     np.save('Q', Q)
#             #     break
#             world.visualize()
#             time.sleep(0.5)
#             Q_last = Q.copy()
#             s_t = s_tp1
#             a_t = a_tp1
#             t += 1
#     except KeyboardInterrupt:
#         pass
#     np.save('Q', Q)
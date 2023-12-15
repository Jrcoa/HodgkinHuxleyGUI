import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from math import exp
from scipy.integrate import odeint  

am = lambda vm: (0.1 * (25 - vm)) / (exp((25-vm)/10) - 1)
an = lambda vm: (0.01 * (10 - vm)) / (exp((10-vm)/10) - 1)
ah = lambda vm: 0.07 * exp(-vm/20)
bn = lambda vm: 0.125 * exp(-vm/80)
bm = lambda vm: 4 * exp(-vm/18)
bh = lambda vm: 1/((exp((30-vm)/10) + 1))
CONFIG =  {'i' : 6.2, 'gk' : 36, 'gna' : 120, 'gl' : 0.3, 'vk' : -12, 'vna' : 115, 'vl' : 10.6, 'cm' : 1e-6}


    


class Solver:
    
    def __init__(self, v0=0):
        self.init_state = self.config_init_state(v0)
        self.config = CONFIG

        
        self.bashforth_step_routines = [lambda h, t, st_n : st_n + h * self.NMT_SYSTEM(st_n, t),
                                        lambda h, t, st_n, st_n1 : st_n1 + h * ((3/2) *  self.NMT_SYSTEM(st_n1,t) - (1/2) *  self.NMT_SYSTEM(st_n,t)),
                                        lambda h, t, st_n, st_n1, st_n2 : st_n2 + h * ((23/12) *  self.NMT_SYSTEM(st_n2,t) - (16/12) *  self.NMT_SYSTEM(st_n1,t) + (5/12) *  self.NMT_SYSTEM(st_n,t)), 
                                        lambda h, t, st_n, st_n1, st_n2, st_n3 : st_n3 + h * ((55/24) *  self.NMT_SYSTEM(st_n3,t) - (59/24) *  self.NMT_SYSTEM(st_n2,t) + (37/24) *  self.NMT_SYSTEM(st_n1,t) - (9/24) *  self.NMT_SYSTEM(st_n,t))]

    def config_init_state(self, v0):
        return np.array([v0, an(v0) / (an(v0) + bn(v0)), am(v0) / (am(v0) + bm(v0)), 
                     ah(v0)/ (ah(v0)+bh(v0))])

    # This is an object that the GUI will use to solve the system
    def solve_with_numpy(self, tmax=30):
        times = np.linspace(0, tmax, 100000)
        V_m, n,m,h = tuple(zip(*odeint(self.NMT_SYSTEM, self.init_state, times))) #args=self.config)))
        #V_m = V_m[0]
        V_m = np.array(V_m)
        V_m -= np.ones(len(V_m))*70
        return times, V_m
    
    def solve_with_adams_bashforth(self, tmax=30, s=3):
        dt = 0.02 # higher s can use better dt
        future = [self.init_state]
        time = 0
        T = [time]
        while time < tmax:
            res = self.bashforth_step(future[-1], time, s, dt)
            time += dt * s
            if np.any(np.isnan(res)):
                raise Exception("Value exploded")
            future.append(res)
            T.append(time)
            
        
        V_m, n, m, h = tuple(zip(*future))
        V_m = np.array(V_m)
        V_m -= np.ones(V_m.size)*70

        return T,V_m
    
    def solve_with_rk4(self, tmax=30):
        dt = 0.01
        time = 0
        T = [time]
        states = [self.init_state]
        while time < tmax:
            states.append(self.rk4_step(states[-1], time, dt))
            time += dt
            T.append(time)
        V_m, n, m, h = tuple(zip(*states))
        V_m = np.array(V_m)
        V_m -= np.ones(V_m.size)*70

        return T, V_m

    def set_config(self, config):
        self.config = config
        
    def set_init_state(self, init_state):
        self.init_state = init_state

    def NMT_SYSTEM(self, state, t=None):#, i=0, gk=0, gna=0, gl=0, vk=0, vna=0, vl=0, cm=0):
        """Right hand side of the n,m,h system of H-H model

        Args:
            state (_type_): _description_
            a_n (_type_): _description_
            b_n (_type_): _description_
            a_m (_type_): _description_
            b_m (_type_): _description_
            a_h (_type_): _description_
            b_h (_type_): _description_

        Returns:
            _type_: _description_
        """
        #i = I(t)
        vm = state[0]
        n = state[1]
        m = state[2]
        h = state[3]
        
        #i = self.I(t)
        # state is [n, m, h]
        return np.array([
                (self.config['i'] - self.config['gk'] * (n**4) * (vm - self.config['vk']) - self.config['gna'] * (m**3) * h * (vm - self.config['vna']) - self.config['gl'] * (vm - self.config['vl']))/self.config['cm'],
                an(vm) * (1-n) - bn(vm) * n,
                am(vm) * (1-m) - bm(vm) * m,
                ah(vm) * (1-h) - bh(vm) * h,
                ])

    def rk4_step(self, yold, told, DT):
        k1 = self.NMT_SYSTEM(yold, told)
        k2 = self.NMT_SYSTEM(yold + 0.5 * DT * k1, told + 0.5 * DT)
        k3 = self.NMT_SYSTEM(yold + 0.5 * DT * k2, told + 0.5 * DT)
        k4 = self.NMT_SYSTEM(yold + 1.0 * DT * k3, told + 1.0 * DT, )
        ynew = yold + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return ynew

            
    def bashforth_step(self, state, t, s, DT):
        # makes s steps according to the bashforth algorithm, assuming starting at yn
        states = [state]
        for i in range(s):
            # this line executes the i'th routine with all of the previous states seen as arguments. Note that s_i+1 has one more argument than s_i
            states.append(self.bashforth_step_routines[i](DT, t, *states))
        return states[-1]
    
    #@staticmethod
    def I(self, t):
        if 0 <= t < 5:
            return 0
        else:
            return 50
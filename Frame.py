''' 
Nicolò Pollini, June 2023
Haifa, Israel
'''

# Import packages and modules
import numpy as np
import math
import scipy.io as sio


# Structural analysis class       
class FE_FRAME_2D:
    
    def __init__(self,dir,Nd,cd):
        Mtemp = sio.loadmat(dir+'M.mat')
        Ktemp = sio.loadmat(dir+'K.mat')
        Htemp = sio.loadmat(dir+'H.mat')
        etemp = sio.loadmat(dir+'e.mat')
        self.M = Mtemp['M'] # mass matrix
        self.K = Ktemp['K'] # stiffness matrix
        self.H = Htemp['H'] # transofrmation matrix
        self.e = etemp['e'] # load distribution vector
        self.xi = 5.0/100
        self.Ndamper = Nd
        self.cdamp = cd # kNs/m, Maximum available damping coefficient
        self.Ndof_u, self.Ndof_d = np.shape(self.H) # Number of degrees of freedom in displacement and drift cooridnates
        
    def load(self, dir, filename):
        Ptemp = sio.loadmat(dir + filename + '.mat')
        self.P = Ptemp[filename]
        self.Dt = self.P[0,1] - self.P[0,0]
        self.time = self.P[0]
        self.acc = self.P[1]
        self.Nstep = len(self.time)
    
    def calc_eigen(self):
        matMK = np.dot(np.linalg.inv(self.M),self.K)
        w, v = np.linalg.eig(matMK)
        self.omega = np.sort(np.sqrt(w))
        self.period = math.pi*2/self.omega
        a0 = self.xi*((2.0*self.omega[0]*self.omega[1])/(self.omega[0]+self.omega[1]))
        a1 = self.xi*(2.0/(self.omega[0]+self.omega[1]))
        self.Cs = a0*self.M + a1*self.K

# #########################################################################
# This code was written by Nicolò Pollini,                                %
# Technion - Israel Institute of Technology                               %  
#                                                                         %
#                                                                         %
# Contact: nicolo@technion.ac.il                                          %
#                                                                         %
# Code repository: https://github.com/pollinico/SAND_opt_FVDs             %
#                                                                         %
# Disclaimer:                                                             %
# The author reserves all rights but does not guarantee that the code is  %
# free from errors. Furthermore, the author shall not be liable in any    %
# event caused by the use of the program.                                 %
# #########################################################################
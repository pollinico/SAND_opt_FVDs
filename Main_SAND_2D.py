''' 
Nicolò Pollini, June 2023
Haifa, Israel
'''

# Import packages
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 13})
import gurobipy as gp
import Frame as fr
import scipy.sparse as sp
import time 
from scipy.interpolate import interp1d
import sys 
import os

class Logger(object):
    def __init__(self, myFile):
        self.terminal = sys.stdout
        self.log = open(myFile, "a")
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass 

# Results folder
resFolder = './continuous_sol/'

# Direct the screen output to a file
stdoutOrigin = sys.stdout 
try: # I check if the file exists already, if yes I delte it
    os.remove(resFolder+"output_cont.txt")
except OSError:
    pass

sys.stdout = Logger(resFolder+"output_cont.txt")

# Functions
def diagOne(n,i):
    mat = np.zeros((n,n))
    mat[i,i] = 1.
    return mat

objHist = np.array([])
objBndHist = np.array([])

def mycallback(model, where):
    # https://www.gurobi.com/documentation/9.5/examples/cb_py.html
    global  objHist, objBndHist
    if where == gp.GRB.Callback.MIP:
        cur_obj = model.cbGet(gp.GRB.Callback.MIP_OBJBST)
        cur_bd = model.cbGet(gp.GRB.Callback.MIP_OBJBND)
        objHist = np.append(objHist, cur_obj)
        objBndHist = np.append(objBndHist, cur_bd)
    '''elif where == gp.GRB.Callback.BARRIER:
        primobj = model.cbGet(gp.GRB.Callback.BARRIER_PRIMOBJ)
        objHist = np.append(objHist, primobj)'''


# Initialize Str object
beta  = 1/4. # Newmark's parameters
gamma = 1/2.
dmax = 9.0 # mm, max drift
Nd = 2 # Number of dampers
cdMax = 3.0 # Max damping coefficient in kNs/mm
Str = fr.FE_FRAME_2D('./2dof_ex/', Nd, cdMax)
uMax = gp.GRB.INFINITY #2*dmax
vMax = gp.GRB.INFINITY #500.0 # gp.GRB.INFINITY #
aMax = gp.GRB.INFINITY #12000.0
CdMax = gp.GRB.INFINITY #2*cdMax
fdMax = gp.GRB.INFINITY #200.0

# Load ground acceleration record:
Str.load('./2dof_ex/', 'LA02')

# refine and interpolate time and ground acceleration
tMax = 1e2 # s, max time that I want
tOld   = Str.time[Str.time<=tMax]
accOld = Str.acc[Str.time<=tMax]
fInterp = interp1d(tOld, accOld, kind='cubic')
dt = 0.02
tNew   = np.arange(tOld[0], tOld[-1], dt)
accNew = fInterp(tNew)

# Calculate eigenvalues and Rayleigh damping matrix Cs:
Str.calc_eigen()

# If I want to switch from drift coordinates to displacements:
Str.M  = np.dot(Str.H.T,np.dot(Str.M,Str.H))
Str.K  = np.dot(Str.H.T,np.dot(Str.K,Str.H))
Str.Cs = np.dot(Str.H.T,np.dot(Str.Cs,Str.H))

M  = Str.M / 1e3
Cs = Str.Cs / 1e3
K  = Str.K / 1e3 
e  = Str.e
P =  - np.dot(M, e * accNew * 1e3)

plt.figure()
plt.plot(tNew, accNew*1e3, label='$a_g(t)$')
plt.xlabel("$t$ [$s$]")
plt.ylabel("$a_g$ [$mm/s^2$]")
plt.legend(loc = 'upper right')
plt.savefig(resFolder+'LA02_2d_continuous.png', bbox_inches="tight")

Ndof_d = Str.Ndof_d
Ndof_u  = Str.Ndof_u
Nstep   = int(P.shape[1])
Ndamper = Str.Ndamper

# Create a new model
mod = gp.Model(name='Damp_Opt_SAND')

# Create variables
xd = mod.addMVar(shape=Ndamper, vtype=gp.GRB.CONTINUOUS, lb=0., ub=1., name="xd") # dampers' coeffcients, normalized [0,1]
xu = mod.addMVar(shape=(Ndof_u,Nstep), vtype=gp.GRB.CONTINUOUS, lb=-uMax, ub=uMax, name="xu") # structural displacements
xv = mod.addMVar(shape=(Ndof_u,Nstep), vtype=gp.GRB.CONTINUOUS, lb=-vMax, ub=vMax, name="xv") # structural velocities
xa = mod.addMVar(shape=(Ndof_u,Nstep), vtype=gp.GRB.CONTINUOUS, lb=-aMax, ub=aMax, name="xa") # structural accelerations
Cd = mod.addMVar(shape=(Ndof_u,Ndof_u), vtype=gp.GRB.CONTINUOUS, lb=-CdMax, ub=CdMax, name="Cd") # damping matrix
Fd = mod.addMVar(shape=(Ndof_u,Nstep), vtype=gp.GRB.CONTINUOUS, lb=-fdMax, ub=fdMax, name="Fd") # dampers' forces
d  = mod.addMVar(shape=(Ndof_d,Nstep), vtype=gp.GRB.CONTINUOUS, lb=-dmax, ub=dmax, name="d") # structural drifts

mod.update()

# Set objective
mod.setObjective(xd.sum(), gp.GRB.MINIMIZE)
mod.setParam('OutputFlag', 1)

CdMat = mod.addConstrs( ( \
    Cd[j,k] == \
        sum( cdMax * np.dot(Str.H[:,j].transpose(),np.dot(diagOne(Ndamper,i),Str.H[:,k])) * xd[i] for i in range(Ndamper) ) for j,k in [[_i, _j] for _i in range(Ndof_u) for _j in range(Ndof_u)]), \
            name='Cd_matrix')

u0 = mod.addConstrs((xu[:,i] == np.zeros(Ndof_u) for i in [0]), name='u0')
v0 = mod.addConstrs((xv[:,i] == np.zeros(Ndof_u) for i in [0]), name='v0')
a0 = mod.addConstrs((xa[:,i] == np.dot(np.linalg.inv(M),(P[:,0]-Cs.dot(np.zeros(Ndof_u))-K.dot(np.zeros(Ndof_u)))) for i in [0]), name='a0')

acc_Newmark = mod.addConstrs((xa[:,i+1] ==  1/(beta*dt**2) * (xu[:,i+1]-xu[:,i]) - 1/(beta*dt) * xv[:,i] - (1/(2*beta)-1) * xa[:,i] for i in range(Nstep-1)), name='acc_Newmark')
vel_Newmark = mod.addConstrs((xv[:,i+1] ==  gamma/(beta*dt) * (xu[:,i+1]-xu[:,i]) + (1-gamma/beta) * xv[:,i] + dt*(1-gamma/(2*beta)) * xa[:,i] for i in range(Nstep-1)), name='vel_Newmark')

for i in range(Nstep):
    drifts = mod.addConstr(d[:,i] ==  Str.H @ xu[:,i] , name='drifts')

for i in range(Nstep):
    for j in range(Ndof_u):
        damper_force = mod.addConstr(Fd[j,i] == sum(Cd[j,k] * xv[k,i] for k in range(Ndof_u)) , name='Damper_force')

equilibrium = mod.addConstrs( (M @ xa[:,i] + Cs @ xv[:,i] + Fd[:,i] + K @ xu[:,i] == P[:,i] for i in range(1,Nstep)), name='equilibrium')

# Gurobi solver parameters (cases don't matter, also underscore, i.e. NonConvex == nonconvex == NON_CONVEX https://www.gurobi.com/documentation/8.1/refman/python_parameter_examples.html#PythonParameterExamples)
mod.params.NonConvex = 2
mod.params.NodeMethod = 2 # barrier, -1 automatic - https://www.gurobi.com/documentation/9.5/refman/nodemethod.html#parameter:NodeMethod
mod.params.Method = 2 # barrier, -1 automatic - https://www.gurobi.com/documentation/9.5/refman/method.html#parameter:Method
mod.params.Presolve = 2 # aggressive, -1 automatic - https://www.gurobi.com/documentation/9.5/refman/presolve.html#parameter:Presolve
mod.params.FeasibilityTol = 1e-5
mod.params.OptimalityTol = 1e-5
mod.params.MIPGap = 1 / 100

# Verify model formulation
mod.write(resFolder+'Damp_Opt_SAND.lp')

start = time.time()

# Optimize model
mod.optimize(mycallback)

end = time.time()

obj = mod.getObjective()

print('Optimization finished')
print('Final objective function: ', np.around(cdMax*obj.getValue(), 3) )
print('Optimized damping coefficients [kNs/mm]: ', np.around(cdMax*xd.X, 3) )
print('Max drift [mm]: ', np.max(np.absolute(np.around(d.X, 3))) )
print('Max displacement [mm]: ', np.max(np.absolute(np.around(xu.X, 3))) )
print('Max velocity [mm/s]: ', np.max(np.absolute(np.around(xv.X, 3))) )
print('Max acceleration [mm/s^2]: ', np.max(np.absolute(np.around(xa.X, 3))) )
print('Max damper force [kNs/mm]: ', np.max(np.absolute(np.around(Fd.X, 3))) )
print('Elapsed time [min]: ', np.around((end - start)/60.0, 3) )

for i in range(mod.SolCount):
    mod.Params.SolutionNumber = i
    mod.write(resFolder+f"{i}.sol")

# Print model
print(" ")
print("-----")
print("-----")
print("Print model:")
print(mod)

plt.figure()
plt.plot(tNew, d.X[0,:], alpha=0.75, label='$d_1(t)$')
plt.plot(tNew, d.X[1,:], alpha=0.75, label='$d_2(t)$')
plt.plot([tNew[0], tNew[-1]], [dmax, dmax],'k--')
plt.plot([tNew[0], tNew[-1]], [-dmax, -dmax],'k--')
plt.xlabel("$t$ [$s$]")
plt.ylabel("$d$ [$mm$]")
plt.legend(loc = 'upper right')
plt.savefig(resFolder+'2d_continuous_d.png', bbox_inches="tight")

plt.figure()
plt.plot(tNew, xu.X[0,:], alpha=0.75, label='$u_1(t)$')
plt.plot(tNew, xu.X[1,:], alpha=0.75, label='$u_2(t)$')
plt.xlabel("$t$ [$s$]")
plt.ylabel("$u$ [$mm$]")
plt.legend(loc = 'upper right')
plt.savefig(resFolder+'2d_continuous_u.png', bbox_inches="tight")

plt.figure()
plt.plot(tNew, xv.X[0,:]/1e3, alpha=0.75, label='$\dot{u}_1(t)$')
plt.plot(tNew, xv.X[1,:]/1e3, alpha=0.75, label='$\dot{u}_2(t)$')
plt.xlabel("$t$ [$s$]")
plt.ylabel("$\dot{u}$ [$m/s$]")
plt.legend(loc = 'upper right')
plt.savefig(resFolder+'2d_continuous_v.png', bbox_inches="tight")

plt.figure()
plt.plot(tNew, xa.X[0,:]/1e3, alpha=0.75, label='$\ddot{u}_1(t)$')
plt.plot(tNew, xa.X[1,:]/1e3, alpha=0.75, label='$\ddot{u}_2(t)$')
plt.xlabel("$t$ [$s$]")
plt.ylabel("$\ddot{u}$ [$m/s^2$]")
plt.legend(loc = 'upper right')
plt.savefig(resFolder+'2d_continuous_a.png', bbox_inches="tight")

plt.figure()
plt.plot(d.X[0,:], Fd.X[0,:], alpha=0.75, label='$F_{d1}(d(t))$')
plt.plot(d.X[1,:], Fd.X[1,:], alpha=0.75, label='$F_{d2}(d(t))$')
plt.xlabel("$d$ [$mm$]")
plt.ylabel("$F_d$ [$kN$]")
plt.legend(loc = 'upper right')
plt.savefig(resFolder+'2d_continuous_fd.png', bbox_inches="tight")

plt.figure()
plt.plot(objHist, alpha=0.75, label='$obj$')
plt.legend(loc = 'upper right')
plt.savefig(resFolder+'2d_continuous_obj.png', bbox_inches="tight")

plt.show()

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
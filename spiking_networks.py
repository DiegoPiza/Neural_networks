#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np 
#import matplotlib.pyplot as plt 
from itertools import islice # import this to slice time within the "for" loop
# parameters
Rm     = 1e6    # resistance (ohm)
Cm     = 2e-8   # capacitance (farad)
taum   = Rm*Cm  # time constant (seconds)
Vr     = -.060  # resting membrane potential (volt)
Vreset = -.070  # membrane potential after spike (volt)
Vth    = -.050  # spike threshold (volt)
Vs     = .020   # spiking potential (volt)
dt     = .001   # simulation time step (seconds)
T      = np.int(1)    # total time to simulate (seconds)
time   = np.linspace(dt,T,int(T/dt))

def initialize_simulation():
    # zero-pad membrane potential vector 'V' and spike vector 'spikes'
    V      = np.zeros(time.size) # preallocate vector for simulated membrane potentials
    spikes = np.zeros(time.size) # vector to denote when spikes happen - spikes will be added after LIF simulation
    V[0]   = Vr # set first time point to resting potential
    return V,spikes

def logistic_map(a,x0,nsteps):
    # function to simulate logistic map:
    # x_{n+1} = a * x_n * (1-x_n)
    x    = np.zeros(nsteps)
    x[0] = x0
    for ii in range(1,nsteps):
        x[ii] = a * x[ii-1] * (1-x[ii-1])
    return x

#def plot_potentials(time,V,timeSpikes):
    # plots membrane potential (V) against time (time), and marks spikes with red markers (timeSpikes)
 #  plt.show()
 #  plt.plot(time,V,'k',timeSpikes,np.ones(timeSpikes.size)*Vs,'ro')
 #  plt.ylabel('membrane potential (mV)')
 #  plt.xlabel('time (seconds)')

def check_solutions( result, solution_filename ):
    # check solutions against provided values
    solution = np.load( solution_filename )
    if ( np.linalg.norm( np.abs( result - solution ) ) < 0.1 ):
        print( '\n\n ---- problem solved successfully ---- \n\n' )
def integrate_and_fire( V, spikes, i, Ie ):
    # function to integrate changes in local membrane potential and fire if threshold reached
    # V - vector of membrane potential
    # spikes - spike marker vector
    # i - index (applied to V and spikes) for current time step
    # Ie - input current at this time step (scalar of unit amp)
			
	# 1: calculate change in membrane potential (dV)
    dV= ((Vr-V[i-1])+(Rm*Ie))/taum
	# 2: integrate over given time step (Euler method)
    V[i]=V[i-1]+(dV*dt)
	# 3: does the membrane potential exceed threshold (V > Vth)?
    if V[i]>Vth:
        V[i]=Vreset
        spikes[i]=1
    return V,spikes # output the membrane potential vector and the {0,1} vector of spikes

def problem_2():
    #//////////////////////////////////////////
    # problem 2 - oscillating current input //
    #////////////////////////////////////////
    # Use the LIF implementation from problem 1.
    # Create a current input which:
    #       - starts at 0 A
    #       - oscillates with a cosine of amplitude 20 nA at stim_time[0]
    #       - stops oscillating and returns to 0 A at stim_time[1]
    #
    # output:
    # Plot the resulting simulated membrange potential of the LIF, and save the 
    # membrane potential in a vector named "V_prob2".
    # problem-specific parameters
    stim_time = [.2,.8] # time (seconds) when current turns ON and turns OFF
    f     = 10 # Hz
    phase = np.pi
    # Get x values of the cosine wave= A cos(ωt)
    time   = np.linspace(dt,T,int(T/dt))
    #Creating cosine wave
    wave=2e-8* np.cos(f*2*phase*time-phase)
    #Setting off periods of wave
    wave[0:int(stim_time[0]*1000)]=0
    wave[int(stim_time[1]*1000):]=0

    V,spikes = initialize_simulation() 	# initialize simulation

    for i, t in islice(enumerate(time),1,None): # iterate over each time step
        # input current on
        if t>stim_time[0] and t<stim_time[1]:
            Ie=wave[i]
        else:
            Ie=0
        V, spikes=integrate_and_fire(V,spikes,i,Ie)# iterate over each time step
        
    # add spikes
    V[spikes==1] = Vs
    
    # PLOT membrane potential
    #plot_potentials(time,V,time[spikes==1])
    #plt.title('Problem 2: Oscillating current input')
    
    # output:
    V_prob2 = V
    return V_prob2

def problem_3():
    #////////////////////////////////////////////////////
    # problem 3 - scan across oscillation frequencies //
    #//////////////////////////////////////////////////
    # Using previous problem's simulation (i.e. oscillating current input),
    # run a simulation per frequency stored in the variable "freq".
    # 
    # output: plot the results, and then save the number of spikes generated in 
    # each run in a variable named "nspikes_prob3".
    
    # problem-specific parameters
    stim_time = [.2,.8] # time (seconds) when current turns ON and turns OFF
    freq  = np.linspace(15,50,(50-15)+1) # Hz
    phase = np.pi
    oscillation_amplitude = 4e-8 # amps

    # initialize array
    nSpikes  = np.zeros(freq.size)

    # iterate each freq
    for j,f in enumerate(freq):
        V,spikes = initialize_simulation() 	# initialize simulation
        wave=oscillation_amplitude* np.cos(f*2*phase*time-phase)#creating cosine wave
        #Setting off periods of wave
        wave[0:int(stim_time[0]*1000)]=0
        wave[int(stim_time[1]*1000):]=0
        for i, t in islice(enumerate(time),1,None): # iterate over each time step
                # input current on
                if t>stim_time[0] and t<stim_time[1]:
                    Ie=wave[i]
                else:
                    Ie=0
                V, spikes=integrate_and_fire(V,spikes,i,Ie)# iterate over each time step
        # calculate sum over spikes
        nSpikes[j]=np.sum(spikes)
    
    # PLOT number of spikes per frequency
   # plt.show()
   # plt.plot( nSpikes )
   # plt.title('Problem 3: Scan across oscillating frequencies')
   # plt.xlabel('frequency (Hz)')
   # plt.ylabel('# of spikes')
    
    return nSpikes

def problem_4():
    #//////////////////////////////////////////
    # problem 4 - fluctuating current input //
    #////////////////////////////////////////
    # Use the LIF implementation from simulation 1.
    # Create a current input using a logistic map in the chaotic regime (a=4).
    # Add an additional current step starting at stim_time[0] and ending at 
    # stim_time[1]
    #
    # output:
    # Plot the resulting simulated membrange potential of the LIF, and save the 
    # membrane potential in a vector named "V_prob4".
    
    # Parameters:
    stim_time = [.2,.8] # time (seconds) when current turns ON and turns OFF
    a = 4; lm_x0 = 0.6; lm_range = 5e-8; # parameters for logistic map (ensure the mean is 0 by subtracting 0.5)
    current_step = 1e-8
    current=(logistic_map(4,lm_x0,len(time))-0.5)*lm_range #  logistic map input current
    V,spikes = initialize_simulation() 	# initialize simulation
    for i, t in islice(enumerate(time),1,None): # iterate over each time step
        # input current on
        if t>stim_time[0] and t<stim_time[1]:
            Ie=current[i]+current_step
        else:
            Ie=current[i]
        V, spikes=integrate_and_fire(V,spikes,i,Ie)# iterate over each time step
        
    # add spikes
    V[spikes==1] = Vs
    
    # PLOT membrane potential
   # plot_potentials(time,V,time[spikes==1])
   # plt.title('Problem 4: Chaotic input')

    # output:
    V_prob4 = V
    return V_prob4


#!/usr/bin/env python
# coding: utf-8

# In[97]:


from brian2 import *
import numpy as np 
from itertools import islice # import this to slice time within the "for" loop


# In[98]:


def lif_current_input( I_e ):    
    start_scope()

    # parameters
    taum   = 20*ms          # time constant
    g_L    = 10*nS          # leak conductance
    R_m    = 1/g_L 		# membrane resistance
    E_l    = -70*mV         # leak reversal potential
    Vr     = E_l            # reset potential
    Vth    = -50*mV         # spike threshold
    Vs     = 20*mV          # spiking potential


    # model equations

    eqs = '''
    dv/dt = ( (Vr - v) + (R_m*I_e) ) / taum  : volt (unless refractory)
    '''

    # create neuron
    N = NeuronGroup( 1, model=eqs, threshold='v>Vth', reset='v=Vr', refractory='5*ms', method='euler' )
    # create model
    # initialize model
    N.v = Vr
    # record model state
    M = StateMonitor( N, 'v', record=True )
    S= SpikeMonitor(N)
    run(1000*ms)
    fr=S.num_spikes/(1*second) # Firing rate in hz
    cv=std(diff(S.spike_trains()[0]))/mean(diff(S.spike_trains()[0]))
    # return values
    return fr,cv


# In[101]:


def lif_poisson_input( v_e, v_i, w_e, w_i ):

        start_scope()

        # parameters
        taum   = 20*ms          # time constant
        g_L    = 10*nS          # leak conductance
        E_l    = -70*mV         # leak reversal potential
        E_e    = 0*mV           # excitatory reversal potential
        tau_e  = 5*ms           # excitatory synaptic time constant
        E_i    = -80*mV         # inhibitory reversal potential
        tau_i  = 10*ms          # inhibitory synaptic time constant
        Nin    = 1000	        # number of synaptic inputs
        Ne     = int(0.8*Nin)   # number of excitatory inputs
        Ni     = int(0.2*Nin)   # number of inhibitory inputs
        Vr     = E_l            # reset potential
        Vth    = -50*mV         # spike threshold
        Vs     = 20*mV          # spiking potential

        # model equations
        eqs = '''
        dv/dt = ( E_l - v + g_e*(E_e-v) + g_i*(E_i-v) ) / taum  : volt (unless refractory)
        dg_e/dt = -g_e/tau_e  : 1  # excitatory conductance (dimensionless units)
        dg_i/dt = -g_i/tau_i  : 1  # inhibitory conductance (dimensionless units)
        '''

        # create neuron
        N = NeuronGroup( 1, model=eqs, threshold='v>Vth', reset='v=Vr', refractory='5*ms', method='euler' )

        # initialize neuron
        N.v = E_l

        # create inputs
        Pe = PoissonGroup( 1, (v_e*Ne) ); Pi = PoissonGroup( 1, (v_i*Ni) )

        # create connections
        synE = Synapses( Pe, N, 'w: 1', on_pre='g_e += w_e' ); synE.connect( p=1 ); 
        synI = Synapses( Pi, N, 'w: 1', on_pre='g_i += w_i' ); synI.connect( p=1 ); 

        # record model state
        M = StateMonitor( N, ('v','g_i'), record=True )
        S = SpikeMonitor( N )

        # run simulation
        run( 1000*ms )

        # plot output
        fr=S.num_spikes/(1*second) # Firing rate in hz
        cv=std(diff(S.spike_trains()[0]))/mean(diff(S.spike_trains()[0]))
        #plt.figure(figsize=(15,5)); plt.plot( M.t/ms, M.v[0] );
        # return values
        return fr,cv


# In[111]:


# problem 1
fr_current_input,cv_current_input = lif_current_input( I_e=10*nA );
fr_poisson_input,cv_poisson_input = lif_poisson_input( v_e=10*Hz, v_i=10*Hz, w_e=0.1, w_i=0.4 );

print (cv_current_input)
print(cv_poisson_input)
#How do the two compare? Why do they differ?
#Analysis :The model with constant current has high regularity which is typical of step current inputs,
#this facilitates regular spiking of the model neuron, where the standard deviation of the ISIs 
#is going to be very small, which makes the cv approach 0. For the model with poisson spiking input,
#the post synaptic neuron spikes at highly irregular times, making the standard deviation of the ISIs 
#higher than the previous model, and the cv > 0. The reason why in some poisson simulation results 
#the cv > 1(and not ~1 which would be typical of an ideal poisson process) 
#is because the spikes times in the spike train tend to cluster together, this is commonly 
#described as bursting, this type of pattern is poorly described by a poisson process alone.  


# In[ ]:


#Using the implementation of lif_poisson_input, simulate the model while ve and vi increase together 
#(e.g. v_e = 10 * Hz and v_i = 10 * Hz, then v_e = 20 * Hz and v_i = 20 * Hz).
#For your understanding plot the number of output spikes per second for each simulation as 
#a function of the input rate. How does the output depend on the input? What is 
#an explanation for the observed behavior?

fr= np.linspace(1,20,int(20/1)) #vector with firing rates to evaluate ranging from 5-150 hz in steps of 5
fro=np.zeros(np.size(fr)) #preallocating firing rate of the output
cvo=np.zeros(np.size(fr))#preallocating cv of the output
for i, f in islice(enumerate(fr),1,None): # iterate over each firing rate 
    fro[i],cvo[i] = lif_poisson_input( v_e=f*Hz, v_i=f*Hz, w_e=0.1, w_i=0.4 );

#plt.plot(fr,fro)  

#Analysis:     
# We can observe fluctuations in firing rate with variable input firing frequency (1Hz -20Hz),
#the model neuron firing rate goes to 0 near the 20Hz mark and it stays that way when simulating >20Hz 
#input frequencies. This is mainly due to predominance of inhibitory input in the membrane potential
#given by the higher weights assigned to those synapses. This brings the membrane potential to a
#more negative state, which prevents the post synaptic neuron from ever reaching the threshold 
#for action potential, unless we adjust (increase) the weights or firing rate of only the excitatory input.


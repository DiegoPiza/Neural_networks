#!/usr/bin/env python
# coding: utf-8

# In[1]:


from brian2 import *


# In[172]:


def problem_1(N,trials):  
    d=np.zeros(trials)
    for i in range(0,trials):
        x=rand(N)
        y=rand(N)
        #Normalizing to unit length
        x=x/linalg.norm(x) 
        y=y/linalg.norm(y)
        d[i]=dot(x,y)
    mn=mean(d);
    return mn

#mne=zeros(10000)
#for i in range(1,9999):
#    mne[i-1]=problem_1(i,10)   
#plot(mne)
################################################################################
#Analysis:
# If we run the construct starting on 1D to reach 10000D  we can observe that
#the variability of the mean of the dot product decreases as you increase N


# In[182]:


def problem_2(): 
    start_scope()
    N = 1000
    taum = 10*ms
    taupre = 20*ms
    taupost = taupre
    Ee = 0*mV
    vt = -54*mV
    vr = -60*mV
    El = -74*mV
    taue = 5*ms
    F = 15*Hz
    wmax = .01
    dApre = .01
    dApost = -dApre * taupre / taupost * 1.05
    dApost *= wmax
    dApre *= wmax

    eqs_neurons = '''
    dv/dt = (ge * (Ee-vr) + El - v) / taum : volt
    dge/dt = -ge / taue : 1
    '''

    input = PoissonGroup(N, rates=F)
    neurons = NeuronGroup(1, eqs_neurons, threshold='v>vt', reset='v = vr',
                          method='linear')
    S = Synapses(input, neurons,
                 '''w : 1
                    dApre/dt = -Apre / taupre : 1 (event-driven)
                    dApost/dt = -Apost / taupost : 1 (event-driven)''',
                 on_pre='''ge += w
                        Apre += dApre
                        w = clip(w + Apost, 0, wmax)''',
                 on_post='''Apost += dApost
                         w = clip(w + Apre, 0, wmax)''',
                 )
    S.connect()
    S.w = 'wmax'  #initial weights set to wmax

    s = StateMonitor(neurons, 'v', record=True) 
    # the resulting ndarray will now have the shape (N,timesteps),
    # which will allow accessing the time evolution of individual 
    # synapses (e.g. mon.w[0] will be the time evolution of the first
    # input synapse)

    s_m = SpikeMonitor(neurons)
    run(10*second)
    spike_train=s_m.spike_trains()[0]
    #CV < 2 seconds
    cv_1=std(diff(spike_train[spike_train<=2*second]))/mean(diff(spike_train[spike_train<=2*second]))
    #CV last 2 seconds
    cv_2=std(diff(spike_train[spike_train>=8*second]))/mean(diff(spike_train[spike_train>=8*second])) 
    return cv_1,cv_2
###############################################################
#Analysis:
#The ISIs tend to be very regular starting the simulation; with fixed synaptic weights,
#the presynaptic input tends to be homogeneous. Once the synaptic weights change due to STDP
#learning, as you progress in time, you increase the variability of the synaptic input, which 
#also increases variability in spike timing of the postsynaptic neuron. Because our parameters 
#are set to slightly favour depotentiation, you are also decreasing the amount of spikes that 
#happen towards the end of the simulation. 


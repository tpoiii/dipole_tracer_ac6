"""
ac6 curtains scene radius - pulse radius dependence

source regions at various distances from ac6 and multiple rs
Different pulse radii at each L
ac6b ~60 seconds behind ac6b

Contributors: Colby Lemon, Paul O'Brien

"""

import json
import numpy as np
import matplotlib.pyplot as plt
import dipole_tracer # import so we can set module variables
from dipole_tracer import fake_ac6,SourceRegion,plot_scene,fly_in_scene2,read_ac6_iso_response_csv

dipole_tracer.debug = False
dipole_tracer.N_POOL = 40 # use 40 processes

L0s = [4.95,5,5.05] # L center of source region
E0 = 70 # e-folding energy of source, keV
J0 = 1 # pulse flux
r0s = [150, 75, 20] # transverse dimension of source region
pulse_period = 0.7 # time between pulses, seconds
pulse_width = 0.1 # pulse width, seconds
t0 = 0 # reference pulse start time, seconds
time_shift = 59.75 # time shift between a and b s/c
oversample = 50 # time sample is oversample Hz to get details (50 for fine)

ac6a = fake_ac6(4,8,t0=0,dt=1/oversample)
ac6b = fake_ac6(4,8,t0=time_shift,dt=1/oversample) # behind by 60 seconds

iso = read_ac6_iso_response_csv('data/ac6a_response_iso.csv')
dos1resp = {'keV':iso['MeV']*1e3,'DOS1':iso['DOS1']} # save for later
JkeV = (dos1resp['keV'] >= 30) & (dos1resp['keV']<=300)

plt.close('all')
colors = ['b','r','m','c','darkgreen']
PHI0_DEG = [0,-0.1,-0.2,-0.5,-1.0] # longitude of the center of the source regions
doseratea = [None]*len(PHI0_DEG)
doserateb = [None]*len(PHI0_DEG)
sources = [None]*len(PHI0_DEG)

settings = {'Ls0':L0s,'r0s':r0s,'E0':E0,'J0':J0,'pulse_period':pulse_period,
            'pluse_width':pulse_width,'t0':t0,'time_shift':time_shift,
            'oversample':oversample,'PHI0_DEG':PHI0_DEG,
            'debug':dipole_tracer.debug}
with open('scene-rs-settings.json','w') as fid:
    json.dump(settings,fid,indent=1)


for (iphi0,phi0_deg) in enumerate(PHI0_DEG):
    phi0 = np.radians(phi0_deg)
    regions = [SourceRegion(L0,phi0,r0,E0,J0,pulse_period,pulse_width,t0,color=colors[iphi0]) for (L0,r0) in zip(L0s,r0s)]

    resulta = fly_in_scene2(ac6a,regions,dos1resp['keV'][JkeV],dos1resp['DOS1'][JkeV])
    resultb = fly_in_scene2(ac6b,regions,dos1resp['keV'][JkeV],dos1resp['DOS1'][JkeV])
    doseratea[iphi0] = resulta['rate']
    doserateb[iphi0] = resultb['rate']
    sources[iphi0] = regions

allregions = [region for sublist in sources for region in sublist] # flatten list of lists

#%%
plot_scene(ac6a,allregions,min_L=L0s[0]-0.5,max_L=L0s[-1]+0.5)
plt.title('All Source Regions')
plt.savefig('figures/scene-rs-scene')

Lrange = [L0s[0]-0.03,L0s[-1]+0.075]

iLa = (ac6a['L']>=Lrange[0]) & (ac6a['L']<=Lrange[-1])
iLb = (ac6b['L']>=Lrange[0]) & (ac6b['L']<=Lrange[-1])
plt.figure()
for (iphi0,phi0_deg) in enumerate(PHI0_DEG):
    plt.plot(ac6a['t'][iLa],doseratea[iphi0][iLa],'-',label='AC6-A,${\phi}_s=%g^o$' % phi0_deg,color=colors[iphi0])
    plt.plot(ac6b['t'][iLb]-time_shift,doserateb[iphi0][iLb],'--',label='AC6-B,${\phi}_s=%g^o$' % phi0_deg,color=colors[iphi0])

xrange = ac6a['t'][iLa][[0,-1]] # extent of time series plot
it0 = [np.argmin(np.abs(ac6a['L']-L0)) for L0 in L0s]
for (r0,L0,it) in zip(r0s,L0s,it0):
    plt.text(ac6a['t'][it],-0.002,'r$_s$=%g km' % (r0),ha='center',va='top')
    # add horizontal lines through centers of steady-state rate
    y = doseratea[-1][it]
    plt.plot(xrange,[y,y],'k--')
    plt.text(58,y,'%.3f' % y,ha='center',va='bottom')

for source in sources[-1]:
    hss = plt.plot(ac6a['t'][iLa],source.steady_state(ac6a['L'][iLa],ac6a['theta'][iLa],dosResp=dos1resp),'k-')
hss[0].set_label('$c_{ss}$') # set label on last one only

plt.ylim([-0.02,0.25])

plt.xlabel('t, seconds (AC6-B shifted by %g seconds)' % time_shift)
plt.ylabel('DOS1 rate')
leg = plt.legend(fontsize='medium',loc='upper right')
plt.savefig('figures/scene-rs-rates')

plt.xlim([ac6a['t'][iLa][0],ac6a['t'][it0[-1]]+1])
leg.remove()
plt.savefig('figures/scene-rs-rates-noleg')

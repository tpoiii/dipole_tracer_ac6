"""
ac6 curtains scene E0 - energy dependence

source regions at various distances from ac6 and multiple Ls
Different E0 at each L
ac6b ~60 seconds behind ac6b

"""

import json
import numpy as np
import matplotlib.pyplot as plt
import dipole_tracer # import so we can set module variables
from dipole_tracer import fake_ac6,SourceRegion,plot_scene,fly_in_scene2,read_ac6_iso_response_csv,get_energy_limits, Bmag
from odc_util import BouncePeriod, DriftPeriod


dipole_tracer.debug = False
dipole_tracer.N_POOL = 40 # use 40 processes

L0s = [4.95,5,5.05] # L center of source region
E0s = [20,70,150] # e-folding energy of source, keV
J0s = [None]*len(E0s) # pulse flux, scales up low energy
r0 = 75 # transverse dimension of source region
pulse_period = 0.7 # time between pulses, seconds
pulse_width = 0.1 # pulse width, seconds
t0 = 0 # reference pulse start time, seconds
time_shift = 59.75 # time shift between a and b s/c
oversample = 50 # time sample is oversample Hz to get details (50 for fine)

ac6a = fake_ac6(4,8,t0=0,dt=1/oversample)
ac6b = fake_ac6(4,8,t0=time_shift,dt=1/oversample) # behind by 60 seconds

iso = read_ac6_iso_response_csv('data/ac6a_response_iso.csv')
dos1resp = {'keV':iso['MeV']*1e3,'DOS1':iso['DOS1']} # save for later
Emin = np.inf
Emax = -np.inf
Emid = [None]*len(E0s)
# equatorial pitch angle of locally mirroring particle
Lmid = np.median(L0s)
iLmid = np.argmin(np.abs(ac6a['L']-Lmid))
aeq90deg = np.degrees(np.arcsin(np.sqrt(Bmag(Lmid,np.pi/2)/Bmag(Lmid,ac6a['theta'][iLmid]))))
tbounce = [None]*len(E0s)
tdrift = [None]*len(E0s)
for (iE0,E0) in enumerate(E0s):
    elimits = get_energy_limits(dos1resp,E0)
    Emin = np.minimum(Emin,elimits['JEmin'])
    Emax = np.maximum(Emax,elimits['JEmax'])
    Emid[iE0] = elimits['Emid']
    tdrift[iE0] = DriftPeriod('e-',Emid[iE0]/1e3,aeq90deg,Lmid)    
    tbounce[iE0] = BouncePeriod('e-',Emid[iE0]/1e3,aeq90deg,Lmid)    
    # scale J0 to integral of response*flux
    J0s[iE0] = 1/elimits['RJ'][-1] 
JkeV = (dos1resp['keV'] >= Emin) & (dos1resp['keV']<=Emax)

plt.close('all')
colors = ['b','r','m','c','darkgreen']
PHI0_DEG = [0,-0.1,-0.2,-0.5,-1.0] # longitude of the center of the source regions
doseratea = [None]*len(PHI0_DEG)
doserateb = [None]*len(PHI0_DEG)
sources = [None]*len(PHI0_DEG)

settings = {'Ls0':L0s,'r0':r0,'E0s':E0s,'J0s':J0s,'pulse_period':pulse_period,
            'pluse_width':pulse_width,'t0':t0,'time_shift':time_shift,
            'oversample':oversample,'PHI0_DEG':PHI0_DEG,
            'debug':dipole_tracer.debug}
with open('scene-E0-settings.json','w') as fid:
    json.dump(settings,fid,indent=1)


for (iphi0,phi0_deg) in enumerate(PHI0_DEG):
    phi0 = np.radians(phi0_deg)
    regions = [SourceRegion(L0,phi0,r0,E0,J0,pulse_period,pulse_width,t0,color=colors[iphi0]) for (L0,E0,J0) in zip(L0s,E0s,J0s)]
    
    resulta = fly_in_scene2(ac6a,regions,dos1resp['keV'][JkeV],dos1resp['DOS1'][JkeV])
    resultb = fly_in_scene2(ac6b,regions,dos1resp['keV'][JkeV],dos1resp['DOS1'][JkeV])
    doseratea[iphi0] = resulta['rate']
    doserateb[iphi0] = resultb['rate']
    sources[iphi0] = regions

allregions = [region for sublist in sources for region in sublist] # flatten list of lists
    
#%%

plot_scene(ac6a,allregions,min_L=L0s[0]-0.5,max_L=L0s[-1]+0.5)
plt.title('All Source Regions')
plt.savefig('figures/scene-E0-scene')

iLa = (ac6a['L']>=L0s[0]-0.025) & (ac6a['L']<=L0s[-1]+0.1)
iLb = (ac6b['L']>=L0s[0]-0.025) & (ac6b['L']<=L0s[-1]+0.1)
plt.figure()
for (iphi0,phi0_deg) in enumerate(PHI0_DEG):
    plt.plot(ac6a['t'][iLa],doseratea[iphi0][iLa],'-',label='AC6-A,${\phi}_s=%g^o$' % phi0_deg,color=colors[iphi0])
    plt.plot(ac6b['t'][iLb]-time_shift,doserateb[iphi0][iLb],'--',label='AC6-B,${\phi}_s=%g^o$' % phi0_deg,color=colors[iphi0])
    
xrange = ac6a['t'][iLa][[0,-1]] # extent of time series plot
it0 = [np.argmin(np.abs(ac6a['L']-L0)) for L0 in L0s]
for (it,E0,L0,J0,Em,tb,td,source) in zip(it0,E0s,L0s,J0s,Emid,tbounce,tdrift,sources[-1]):
    plt.text(ac6a['t'][it],12,'E$_s$=%g keV\nj$_s$=%.0f' % (E0,J0),ha='center',va='bottom')
    # add horizontal lines through centers of steady-state rate
    iL = np.abs(ac6a['L']-L0)<(L0s[1]-L0s[0])/2
    y = doseratea[-1][iL].max()
    plt.plot(xrange,[y,y],'k--')
    plt.text(58,y,'%.1f (%.0f)' % (y,td/tb/360),ha='center',va='bottom')    

for source in sources[-1]:
    hss = plt.plot(ac6a['t'][iLa],source.steady_state(ac6a['L'][iLa],ac6a['theta'][iLa],dosResp=dos1resp),'k-')
hss[0].set_label('$c_{ss}$') # set label on last one only
    
plt.ylim([0,14])

plt.xlabel('t, seconds (AC6-B shifted by %g seconds)' % time_shift)
plt.ylabel('DOS1 rate')
leg = plt.legend(fontsize='small')
plt.savefig('figures/scene-E0-rates')

plt.xlim([ac6a['t'][it0[0]]-1,ac6a['t'][it0[-1]]+1])
leg.remove()
plt.savefig('figures/scene-E0-rates-noleg')

"""
ac6 curtains scene real: Figure 1 & 2 from Blake and O'Brien 2016
"""

from sys import argv
import json
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
#import matplotlib.dates as mdates
import dipole_tracer # import so we can set module variables
from dipole_tracer import makeSourceRegion,SourceCluster,plot_scene,\
    fly_in_scene2,read_ac6_iso_response_csv,ConvMean,gdz2xyz,xyz2state,\
    seedRNG,fake_ac6,get_energy_limits,load_rbsp_plasmapause
from load_ac6_data_np import load_ac6_file
from tictoc import tic,toc
from scipy.interpolate import interp1d,BSpline
from spacepy import pycdf

debug = False
resolution = 'MEDIUM' # LOW, MEDIUM, HIGH
source_method = 'PEAKS' # RATES, PEAKS, CLUSTERS
# RATES - small sources, with number scaled by curtain dose rate
# PEAKS - large source at each "peak" in dose rate
# CLUSTERS - break PEAKS into smaller sources of size ~r0
if len(argv)>1:
    if argv[1] in ['RATES','PEAKS','CLUSTERS']:
        source_method = argv[1]
    else:
        raise Exception('Unknown argument %s' % argv[1])
print('source_method:',source_method)

dipole_tracer.debug = debug
dipole_tracer.N_POOL = 40 # use 40 processes
#VWcache = VWCache('vw_cache.pickle')
#VWcache = VWCache('vw_cache.pickle',local=True)
VWcache = None # final version does not use VWcaching

r0 = 75 # default transverse dimension of source region. Shumko suggests ~75 km in single-fit analysis
E0 = 70 # e-folding energy of source, keV
pulse_period = 0.7 # time between pulses, seconds (don't make it an integer # of seconds)
pulse_width = 0.1 # pulse width, seconds
pulse_frac = pulse_width/pulse_period # fraction 'on' for pulses
#t0 = 0 # default reference pulse start time, seconds (not used)

Lrange = (0,np.inf) # limit tracing to specific L range
bequalsa = False # copy a to b instead of tracing separately. saves half time

if debug:
    resolution = 'LOW'
    #dipole_tracer.N_POOL = 1
    Lrange = (4,5)
    bequalsa = True

if resolution == 'LOW':
    oversample = 10
    Na = 10 # 5,23.88,...175 degree local pitch angle sampling
    Estep = 5 # step size in energy grid
elif resolution == 'MEDIUM':
    oversample = 10 # time sample is oversample Hz to get details (50 for fine)
    Na = 18 # 5,15,...175 degree local pitch angle sampling
    Estep = 2 # step size in energy grid
elif resolution == 'HIGH':
    oversample = 50 # time sample is oversample Hz to get details (50 for fine)
    Na = 100 # 5,6.7,...175 degree local pitch angle sampling
    Estep = 1 # step size in energy grid
else:
    raise Exception('unknown resolution: %s' % resolution)

ac6a_real = load_ac6_file('A', dt.datetime(2015,2,7), 'survey', ac6_dir='./data')
ac6b_real = load_ac6_file('B', dt.datetime(2015,2,7), 'survey', ac6_dir='./data')
rbspa = pycdf.CDF('./data/rbspa_ect-elec-L2_20150207_v2.1.0.cdf')
rbspb = pycdf.CDF('./data/rbspb_ect-elec-L2_20150207_v2.1.0.cdf')
Lppa = load_rbsp_plasmapause('A')
Lppb = load_rbsp_plasmapause('B')

time_shift = 65 # time shift between a and b s/c
dtb = dt.timedelta(seconds=time_shift)
if False:  # Figure 1
    ascale = 2.5 # scaling for ac6a to match ac6b
    ta_range = np.array([dt.datetime(2015,2,7,21,49,30),dt.datetime(2015,2,7,21,54,30)])
    min_L = 3 # min_L of sim and plot
    max_L = 15  # max_L of sim and plot
    bg_func = lambda L: 50*10**(-((L-9)/10)**2)
    sources = [(7.0,1e-9,10,None),(8.3,1e-9,10,None),(9.5,2e-6,2,0)] # (L0,dJ0,dphi0,t0)
else:
    ascale = 4 # scaling for ac6a to match ac6b
    ta_range = np.array([dt.datetime(2015,2,7,20,11,0),dt.datetime(2015,2,7,20,15,00)]) # Figure 2
    min_L = 3.6 # min_L of sim and plot
    max_L = 9.5  # max_L of sim and plot
    bg_func = lambda L: 60*10**(-((L-5.5)/2)**2)+40*10**(-((L-8)/2)**2)

uth0 = ta_range[0].hour + ta_range[1].minute/60 # decimal hour to the minute
tb_range = ta_range+dtb
ita = (ac6a_real['dateTime']>=ta_range[0]) & (ac6a_real['dateTime']<=ta_range[1])
itb = (ac6b_real['dateTime']>=tb_range[0]) & (ac6b_real['dateTime']<=tb_range[1])

# add dipole coordinates
for (ac6,it,color) in [(ac6a_real,ita,'b'),(ac6b_real,itb,'r')]:
    ac6['it'] = it
    ac6['t'] = np.array([(t-ta_range[0]).total_seconds() for t in ac6['dateTime'][it]])
    ac6['color'] = color
    (ac6['x'],ac6['y'],ac6['z']) = gdz2xyz(ac6['alt'][it],ac6['lat'][it],ac6['lon'][it]+uth0*15) # phi is approx local time
    (ac6['L'],ac6['theta'],ac6['phi']) = xyz2state(ac6['x'],ac6['y'],ac6['z'])
    ac6['Xeq'] = ac6['L']*np.cos(ac6['phi'])
    ac6['Yeq'] = ac6['L']*np.sin(ac6['phi'])

# now oversample dipole coordinates to make ac6a,ac6b dicts

ac6a = {}
ac6b = {}
for (fast,real,it) in [(ac6a,ac6a_real,ita),(ac6b,ac6b_real,itb)]:
    fast['t'] = np.linspace(real['t'][0],real['t'][-1],int((real['t'][-1]-real['t'][0])*oversample+1))
    for key in ['alt','lat','lon']:
        fast[key] = interp1d(real['t'],real[key][it])(fast['t'])
    for key in ['x','y','z','L','theta','phi','Xeq','Yeq']:
        fast[key] = interp1d(real['t'],real[key])(fast['t'])

#phi_ac6_deg = 140
#ac6a = fake_ac6(min_L,max_L,t0=0,dt=1.0/oversample,lon=phi_ac6_deg,alt=660)
#ac6b = fake_ac6(min_L,max_L,t0=time_shift,dt=1.0/oversample,lon=phi_ac6_deg,alt=660)

iso = read_ac6_iso_response_csv('data/ac6a_response_iso.csv')
dos1resp = {'keV':iso['MeV']*1e3,'DOS1':iso['DOS1']} # save for later
elimits = get_energy_limits(dos1resp,E0)
print('Energy Filtering from %.1f to %.1f keV, %.1f %% of counts' % (elimits['JEmin'],elimits['JEmax'],elimits['fracRJkeV']*100))
print('Peak energy is %.1f keV, 50%% energy is %.1f keV' % (elimits['Epeak'],elimits['Emid']))
JkeV = elimits['JkeV'] & (np.arange(len(elimits['JkeV'])) % Estep == 0) # decimate energy grid

#%%
plt.close('all')


#%% generate RBSP fluxes
# good apogee at 15:50, perigee at 20:13
rbspa_profile = {}
rbspb_profile = {}
trbsp = [dt.datetime(2015,2,7,15,50),dt.datetime(2015,2,7,20,13)]
for (profile,data,ac6,Lpp) in [(rbspa_profile,rbspa,ac6a_real,Lppa),(rbspb_profile,rbspb,ac6b_real,Lppb)]:
    # compute dipole L from near-equatorial position
    (L,theta,phi) = xyz2state(data['Position'][:].transpose()/dipole_tracer.akm)
    it = ((data['Epoch'][:] >= trbsp[0]) & (data['Epoch'][:] <= trbsp[1]) 
        & (data['L'][:] >= ac6['Lm_OPQ'][ac6['it']].min()) & (data['L'][:] <= ac6['Lm_OPQ'][ac6['it']].max()))
    profile['Epoch'] = data['Epoch'][:][it]
    profile['t'] = np.array([(t-trbsp[0]).total_seconds() for t in profile['Epoch']])    
    profile['it'] = it
    profile['Lm_OPQ'] = data['L'][:][it]
    profile['MLT_OPQ'] = data['MLT'][:][it]
    profile['L_dip'] = L[it]
    profile['rate'] = np.full(profile['Lm_OPQ'].shape,0.0)
    keV = np.array(data['FESA_FIT_Energy'][:])
    for (i,jt) in enumerate(np.nonzero(it)[0]): # i index into profile, jt index into cdf data
        # get energy range based on positive values in FESA_FIT
        y0 = data['FESA_FIT'][jt,:].copy()
        iE = np.nonzero(y0>0)[0]
        if len(iE) == 0:
            continue
        iEmin = keV[iE[0]]
        iEmax = keV[iE[-1]]
        IkeV = (dos1resp['keV'] >= iEmin) & (dos1resp['keV']<=iEmax)
        knots = data['FESA_FIT_Knots'][jt,:]
        coefs = data['FESA_FIT_Coeffs'][jt,:]
        n = np.sum(knots!= -1e31)
        spl = BSpline(knots[0:n],coefs[0:n],3)
        y = 10**spl(np.log10(dos1resp['keV'][IkeV]))
        rate = np.trapz(dos1resp['DOS1'][IkeV]*y,dos1resp['keV'][IkeV])
        profile['rate'][i] = rate        
    iLpp = np.nonzero((Lpp['Inbound'] >= trbsp[0]) & (Lpp['Inbound'] <= trbsp[1]))[0][0]
    profile['Lpp_Epoch'] = Lpp['Inbound'][iLpp]
    profile['Lpp_OPQ'] = interp1d(profile['t'],profile['Lm_OPQ'])((Lpp['Inbound'][iLpp]-trbsp[0]).total_seconds())
    profile['Lpp_dip'] = interp1d(profile['t'],profile['L_dip'])((Lpp['Inbound'][iLpp]-trbsp[0]).total_seconds())
    profile['Lpp_rate'] = interp1d(profile['t'],profile['rate'])((Lpp['Inbound'][iLpp]-trbsp[0]).total_seconds())

plt.figure()
plt.semilogy(ac6a_real['Lm_OPQ'][ita],ac6a_real['dos1rate'][ita]*ascale,ac6a_real['color']+'-',label='AC6-A x %g' % ascale)
plt.semilogy(ac6b_real['Lm_OPQ'][itb],ac6b_real['dos1rate'][itb],ac6b_real['color']+'-',label='AC6-B')
plt.semilogy(rbspa_profile['Lm_OPQ'],rbspa_profile['rate'],'b--',label='DOS1 at RBSP-A')
plt.semilogy(rbspb_profile['Lm_OPQ'],rbspb_profile['rate'],'r--',label='DOS1 at RBSP-B')
plt.semilogy(ac6a_real['Lm_OPQ'][ita],bg_func(ac6a_real['L']),'k--',label='Background') # evaluate on dipole L, plot on Lm_OPQ
ylim = plt.ylim()
plt.semilogy(rbspa_profile['Lpp_OPQ'],rbspa_profile['Lpp_rate'],'bv',label="Plasmapause")
plt.semilogy(rbspb_profile['Lpp_OPQ'],rbspb_profile['Lpp_rate'],'rv')
plt.legend()
plt.ylabel('#/s')
plt.xlabel('$L_{m,OPQ}$')
plt.grid('on')
plt.savefig('figures/scene-real-rbsp-rates.png')

#%%
plt.figure()
h = plt.semilogy(ac6a_real['dateTime'][ita],ac6a_real['dos1rate'][ita]*ascale,ac6a_real['color']+'-',ac6b_real['dateTime'][itb]-dtb,ac6b_real['dos1rate'][itb],ac6b_real['color']+'-')
ac6a_real['bg'] = bg_func(ac6a_real['L'])
ac6b_real['bg'] = bg_func(ac6b_real['L'])
#plt.semilogy(ac6a_real['dateTime'][ita],ac6a_real['bg'],ac6a_real['color']+'--',ac6b_real['dateTime'][itb]-dtb,ac6b_real['bg'],ac6b_real['color']+'--')
plt.xlabel(ta_range[0].strftime('%d %B %Y'))
plt.xlim(ta_range)
plt.ylim([1e1,3e2])
plt.ylabel('DOS1 Rate')
plt.legend(h,['AC6-Ax%g' % ascale,'AC6-B shifted by %g seconds' % time_shift])
ax = plt.gca()
plt.setp(ax.get_xticklabels(), ha="right", rotation=30)
plt.savefig('figures/scene-real-rates.png')



#%%

# make sources based on flux ratio
y = ac6b_real['dos1rate'][itb] / ac6b_real['bg']
x = ac6b_real['L']

# small sources scattered in proportion to flux ratio
rate_sources= []
for i in range(len(x)):
    if not (y[i]>1): # skips on nan or y<=1
        continue # skip this L
    L0 = x[i]
    Ngoal = np.maximum(0,y[i]/1 - 1) # desired number of sources
    Nsources = int(np.maximum(1,np.round(Ngoal))) # integer number of sources
    source = {'L0':L0,'dJ0':1.8/elimits['RJ'][-1]*Ngoal/Nsources*(L0/8)**4.5,'dphi0':[5,20],'cluster':False} 
    # for E0 = 70 keV, cint[-1] = 0.014
    # for E0 = 10 keV, cint[-1] = 5e-5
    rate_sources.extend([source]*Nsources)

# make sources based on peaks: size based on flux ratio
ipeaks = []
peak_sources = []
cluster_sources = []
while np.any(np.isfinite(y)):
    imax = np.nanargmax(y)
    L0 = x[imax]
    ipeaks.append(imax)
    
    # PEAKS source
    source = {'L0':L0,'r0':100*y[imax]*(8/L0),'dJ0':0.4/elimits['RJ'][-1]*(L0/8)**5.5,'dphi0':[5,20],'cluster':False}
    peak_sources.append(source)

    # CLUSTER source
    source = {**source,'cluster':True}
    cluster_sources.append(source)
    
    y[np.abs(x-x[imax])<0.1/2] = np.nan # blank out near peak



#%%
plt.figure()
plt.semilogy(ac6b_real['L'],ac6b_real['dos1rate'][itb],'-',label='AC6-B',color=ac6b_real['color'])
plt.semilogy(ac6b_real['L'],ac6b_real['bg'],'k-',label='Background')
plt.semilogy(ac6b_real['L'][ipeaks],ac6b_real['dos1rate'][itb][ipeaks],'go',label='Sources')
plt.xlabel('L')
plt.ylabel('DOS1 Rate')
plt.legend()
plt.savefig('figures/scene-real-sources.png')

#%%
default = {'L0':5,'dJ0':2e-10,'dphi0':5,'t0':None,'pulse_frac':pulse_frac,'pulse_period':pulse_period,'r0':r0,'cluster':False}
# L0 - center of source region
# dJ0 - flux over background
# dphi0 is azimuth offset (>=0) from ac6 in deg, or [low,high] random range
# t0 in seconds None-random
# pulse_frac - fraction of time source is on
# pulse_period - pulse period in seconds
# cluster - break source into cluster?
if source_method == 'RATES':
    sources = rate_sources
elif source_method == 'PEAKS':
    sources = peak_sources
elif source_method == 'CLUSTERS':
    sources = cluster_sources
else:
    raise Exception('Unknown source_method %s' % source_method)
sources = list(filter(lambda x:(x['L0']>=Lrange[0] and x['L0']<=Lrange[1]),sources))

colors = ['b','r','m','c','orange','darkgreen','brown']

regions = []
randomState = seedRNG() # reset random number generator to default seed

for (isource,source) in enumerate(sources):
    source = {**default,**source}    
    J0 = bg_func(source['L0'])*source['dJ0']
    dphi0 = source['dphi0']
    if not np.isscalar(dphi0): # roll random phase degrees
        dphi0 = randomState.random()*(dphi0[1]-dphi0[0])+dphi0[0]
    phi0 = interp1d(ac6a['L'],ac6a['phi'],fill_value='extrapolate',bounds_error=False)(source['L0'])-np.radians(dphi0)
    pw = source['pulse_period']*source['pulse_frac']
    color = colors[isource % len(colors)]
    if source['cluster']:
        regions.extend(SourceCluster(source['L0'],phi0,source['r0'],r0,E0,J0,source['pulse_period'],pw,t0=source['t0'],color=color,type='gaussian'))
    else:
        regions.append(makeSourceRegion(source['L0'],phi0,source['r0'],E0,J0,source['pulse_period'],pw,t0=source['t0'],color=color,type='gaussian'))

settings = {**default,'E0':E0,'time_shift':time_shift,
            'source_method':source_method,'resolution':resolution,
            'Lrange':Lrange,'debug':debug,'bequalsa':bequalsa}
settings['regions'] = [r.toDict() for r in regions]

with open('scene-real-settings-%s.json' % source_method,'w') as fid:
    json.dump(settings,fid,indent=1)


#%% make fake AC6-A to show micorbursts
# AC6-B behind AC6-A as in reality
# fixed-phi trajectories through middle of field
# remove sources that are past trajectory
foversample = 50 # fine oversample to produce 10 Hz data
phi0 = np.median(ac6a['phi'])-np.radians(15)
fac6a = fake_ac6(min_L,max_L,t0=0,dt=1/foversample,lon=np.degrees(phi0))
fac6b = fake_ac6(min_L,max_L,t0=time_shift,dt=1/foversample,lon=np.degrees(phi0)) # behind by 60 seconds
fac6a['color'] = ac6a_real['color']
fac6b['color'] = ac6b_real['color']
fregions = list(filter(lambda x:(x.phi0<=phi0),regions))

#%%
plot_scene(ac6a,regions,min_L=min_L,max_L=max_L,trace_shells=False)
for ac6 in [ac6a_real,ac6b_real]:
    iL = (ac6['L'] >= min_L) & (ac6['L'] <= max_L)
    plt.plot(ac6['Xeq'][iL],ac6['Yeq'][iL],ac6['color']+'-')
plt.plot(fac6a['Xeq'],fac6a['Yeq'],'k--')
plt.savefig('figures/scene-real-scene-%s.png' % source_method)

# raise Exception

#%% trace the scene at real AC6 location
tmain = tic()
resulta = fly_in_scene2(ac6a,regions,dos1resp['keV'][JkeV],dos1resp['DOS1'][JkeV],VWcache=VWcache,Na=Na)
if bequalsa:
    resultb = resulta ; print('DEBUG!')
else:
    resultb = fly_in_scene2(ac6b,regions,dos1resp['keV'][JkeV],dos1resp['DOS1'][JkeV],VWcache=VWcache,Na=Na)
toc(start=tmain)

if VWcache:
    VWcache.save()

# compute 1-second averages for comparison to figure
a1sec = {}
b1sec = {}
for (low,fake,res) in [(a1sec,ac6a,resulta),(b1sec,ac6b,resultb)]:
    low['t'] = fake['t'][oversample//2::oversample]
    low['L'] = fake['L'][oversample//2::oversample]
    low['raw'] = ConvMean(res['rate'],1-oversample//2,oversample//2)[oversample//2::oversample]
    low['rate'] = low['raw'] + bg_func(low['L'])   

#%%
plt.figure()
plt.semilogy(ac6b_real['t']-time_shift,ac6b_real['dos1rate'][itb],color=[0.5,0.5,0.5],label='Real AC6-B')
plt.semilogy(a1sec['t'],a1sec['rate'],'-',label='Sim AC6-A',color='b')
plt.semilogy(b1sec['t']-time_shift,b1sec['rate'],'--',label='Sim AC6-B',color='r')
plt.xlabel('t, seconds (AC6-B shifted by %g seconds)' % time_shift)
plt.ylabel('DOS1 Rate')
plt.legend()
plt.savefig('figures/scene-real-sim-vs-time-%s.png' % source_method)

plt.figure()
plt.semilogy(ac6b_real['L'],ac6b_real['dos1rate'][itb],'-',label='Real-AC6-B',color=[0.5,0.5,0.5])
plt.semilogy(a1sec['L'],a1sec['rate'],'-',label='Sim AC6-A',color='b')
plt.semilogy(b1sec['L'],b1sec['rate'],'--',label='Sim AC6-B',color='r')
plt.xlabel('L')
plt.ylabel('DOS1 Rate')
plt.legend()
plt.savefig('figures/scene-real-sim-vs-L-%s.png' % source_method)

if debug:
    raise Exception
    pass
    
#%% fake flight through source region

# find the distance in phi of each source to the trajectory
dphi = [interp1d(fac6a['L'],fac6a['phi'],fill_value='extrapolate')(r.L0)-r.phi0 for r in fregions]
imin_source = np.argmin(np.abs(dphi)) # index of closest source  
 
if False: # only include region closest to trajectory
    fregions = [fregions[imin_source]]
    
plot_scene(fac6a,regions,min_L=min_L,max_L=max_L,trace_shells=False)
for ac6 in [fac6a,fac6b]:
    plt.plot(ac6['Xeq'],ac6['Yeq'],ac6['color']+'-')
plt.savefig('figures/scene-real-fake-scene-%s.png' % source_method)

#%
tmain = tic()
fresulta = fly_in_scene2(fac6a,fregions,dos1resp['keV'][JkeV],dos1resp['DOS1'][JkeV],VWcache=VWcache,Na=Na)
if bequalsa:
    fresultb = fresulta ; print('DEBUG!')
else:
    fresultb = fly_in_scene2(fac6b,fregions,dos1resp['keV'][JkeV],dos1resp['DOS1'][JkeV],VWcache=VWcache,Na=Na)
toc(start=tmain)

if VWcache:
    VWcache.save()

#% compute averages
# compute 1 Hz averages
fa1sec = {}
fb1sec = {}
for (low,fake,res) in [(fa1sec,fac6a,fresulta),(fb1sec,fac6b,fresultb)]:
    low['t'] = fake['t'][foversample//2::foversample]
    low['L'] = fake['L'][foversample//2::foversample]
    low['raw'] = ConvMean(res['rate'],1-foversample//2,foversample//2)[foversample//2::foversample]
    low['rate'] = low['raw'] + bg_func(low['L'])   

# compute 10 Hz averages
fa10Hz = {}
fb10Hz = {}
tenHz = foversample//10
for (low,fake,res) in [(fa10Hz,fac6a,fresulta),(fb10Hz,fac6b,fresultb)]:
    low['t'] = fake['t'][tenHz//2::tenHz]
    low['L'] = fake['L'][tenHz//2::tenHz]
    low['raw'] = ConvMean(res['rate'],1-tenHz//2,tenHz//2)[tenHz//2::tenHz]
    low['rate'] = low['raw'] + bg_func(low['L'])   

#%%
plt.figure()
plt.semilogy(fa1sec['t'],fa1sec['rate'],'-',label='Sim AC6-A',color='b')
plt.semilogy(fb1sec['t']-time_shift,fb1sec['rate'],'--',label='Sim AC6-B',color='r')
plt.semilogy(fa1sec['t'],bg_func(fa1sec['L']),'k--',label='Background') # evaluate on dipole L, plot on time
plt.xlabel('t, seconds (AC6-B shifted by %g seconds)' % time_shift)
plt.ylabel('DOS1 Rate (1 Hz sampling)')
plt.legend()
plt.savefig('figures/scene-real-fake-vs-time-%s.png' % source_method)

#%
plt.figure()
plt.semilogy(fa10Hz['t'],fa10Hz['rate'],'-',label='Sim AC6-A',color='b')
plt.semilogy(fb10Hz['t']-time_shift,fb10Hz['rate'],'--',label='Sim AC6-B',color='r')
plt.semilogy(fa10Hz['t'],bg_func(fa10Hz['L']),'k--',label='Background') # evaluate on dipole L, plot on time
plt.xlabel('t, seconds (AC6-B shifted by %g seconds)' % time_shift)
plt.ylabel('DOS1 Rate (10 Hz sampling)')
plt.legend()
plt.savefig('figures/scene-real-fake-vs-time-10Hz-%s.png' % source_method)

#%%
plt.figure()
plt.semilogy(fa10Hz['t'],fa10Hz['rate'],'-',label='Sim AC6-A',color='b')
plt.semilogy(fb10Hz['t']-time_shift,fb10Hz['rate'],'--',label='Sim AC6-B',color='r')
plt.semilogy(fa10Hz['t'],bg_func(fa10Hz['L']),'k--',label='Background') # evaluate on dipole L, plot on time

plt.xlabel('t, seconds (AC6-B shifted by %g seconds)' % time_shift)
plt.ylabel('DOS1 Rate (10 Hz sampling)')
if source_method == 'RATES':
    plt.axis([75,83,40,200])
elif source_method == 'PEAKS':
    plt.axis([30,40,15,30])
else: # CLUSTERS
    plt.axis([1,8,7,15])
    
plt.legend()
plt.savefig('figures/scene-real-fake-vs-time-10Hz-zoom-%s.png' % source_method)


# Note: 20-21 UT on 7 Feb 2015 is the recovery phase of a small Dst ~ -40 storm at ~10 UT earlier in the day.
# Kp max was 3+, but at 20-12 UT it was dropping from 2+ to 1+
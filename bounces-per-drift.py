"""
Create plot showing number of bounces per degree of drift vs energy
"""

import numpy as np
import matplotlib.pyplot as plt
from odc_util import BouncePeriod, DriftPeriod 
from dipole_tracer import read_ac6_iso_response_csv,get_energy_limits

keV = 10**np.linspace(1,4,100)
Tb = np.full(keV.shape,np.nan) # bounce period seconds
Td = np.full(keV.shape,np.nan) # drift period seconds

alpha_eq_deg = 45
L = 5

for (i,E) in enumerate(keV):
    Tb[i] = BouncePeriod('e',E/1e3,alpha_eq_deg,L,unit='deg') # E in keV-> MeV
    Td[i] = DriftPeriod('e',E/1e3,alpha_eq_deg,L,unit='deg') # E in keV-> MeV
    
iso = read_ac6_iso_response_csv('data/ac6a_response_iso.csv')
dos1resp = {'keV':iso['MeV']*1e3,'DOS1':iso['DOS1']} # save for later
E0s = [20,70,150] # energies to look at, keV
colors = ['r','g','b']
elimits = [None]*len(E0s)

# make figure showing response weighted by exp(-E/E0)
for (i,E0) in enumerate(E0s):
    elimits[i] = get_energy_limits(dos1resp,E0)

plt.close('all')
fig,axs = plt.subplots(3,1,sharex=True,gridspec_kw={'height_ratios': [1,1,1]})
fsize = fig.get_size_inches()
fsize[1] *= 2
fig.set_size_inches(fsize)


# plot bounces per degree of drift
plt.sca(axs[0])
plt.loglog(keV,Td/Tb/360,'k-',label=r'Bounces per degree of drift @ L=%g,$\alpha_{eq}$=%g$^{o}$' % (L,alpha_eq_deg))
plt.loglog(keV,(keV/keV[0])**-0.5*Td[0]/Tb[0]/360,'k--') # low energy E dependence 1/sqrt(E)
plt.loglog(keV,(keV/keV[-1])**-1*Td[-1]/Tb[-1]/360,'k--') # high energy E dependence 1/E

plt.ylim([0.5,1e2])
plt.text(30,50,'$E^{-1/2}$')
plt.text(3e3,1,'$E^{-1}$')
plt.ylabel('Ratio')
plt.grid('on')
plt.legend(fontsize='small')

# plot weighted energy response
plt.sca(axs[1])
ymax = 1 # y value at which to plot peak of weighted energy response
for (i,E0) in enumerate(E0s):
    plt.loglog(elimits[i]['Epeak'],elimits[i]['Rjpeak']/elimits[i]['Rjpeak']*ymax,'^',color=colors[i])
    plt.loglog(elimits[i]['Emid'],elimits[i]['Rjmid']/elimits[i]['Rjpeak']*ymax,'o',color=colors[i])
    plt.loglog(dos1resp['keV'],elimits[i]['Rj']/elimits[i]['Rjpeak']*ymax,'-',
               label='E$_s$=%g, max @ %.0f, 50%%$\int$ @ %.0f keV' % (E0,elimits[i]['Epeak'],elimits[i]['Emid']),color=colors[i])

plt.ylim([1e-2,1.2])
plt.ylabel('Weighted Response\n(Arbitrary Units)')
plt.grid('on')
plt.legend(fontsize='small')
t = plt.text(1e3,1.5e-2,'DOS1 response\nweighted by $e^{-E/E_s}$\nnormalized to max = 1')
t.set_bbox({'alpha':0.5,'facecolor':'white','edgecolor':'white'}) # make bg semitransparent

# draw drift and bounce periods
plt.sca(axs[2])
plt.loglog(keV,Td/360,label='Seconds per degree of drift')
plt.loglog(keV,Tb,label='Seconds per bounce')
plt.xlabel('Energy, keV')
plt.legend(fontsize='small')
plt.grid('on')
plt.ylabel('Seconds')

for E in [30, 70, 100, 200, 300, 400, 700]:
    tb = BouncePeriod('e',E/1e3,alpha_eq_deg,L,unit='deg') # E in keV-> MeV
    td = DriftPeriod('e',E/1e3,alpha_eq_deg,L,unit='deg') # E in keV-> MeV
    if E==100:
        va = 'top'
        ha = 'right'
    elif E == 300:
        va = 'top'
        ha = 'center'
    else:
        va = 'bottom'
        ha = 'center'
    axs[2].plot(E,tb,'k.')
    axs[2].annotate('%.2f' % tb,(E,tb),fontsize='small',ha=ha,va=va)
    axs[2].plot(E,td/360,'k.')
    axs[2].annotate('%.0f' % (td/360),(E,td/360),fontsize='small',ha='center',va='bottom')
    axs[0].plot(E,td/tb/360,'k.')
    axs[0].annotate('%.0f' % (td/tb/360),(E,td/tb/360),fontsize='small',ha='right',va='top')

plt.savefig('figures/bounces-per-drift.png')

# analysis
# from odc_util:
#    f = (3*L/2/np.pi/gamma)*(gamma**2-1)*(c/a)**2*(m0*c/abs(q)/B0)*(Dy/Ty)/c # extra 1/c for SI
#    Td = 1.0/f # seconds
#    Tb = 4*L*a/v*Ty
# Tb/Td = Tb*f = (3*L/2/np.pi/gamma)*(gamma**2-1)*(c/a)**2*(m0*c/abs(q)/B0)*(Dy/Ty)/c*4*L*a/v*Ty
# Energy dependence only is:
# Tb/Td ~ (gamma**2-1)/gamma/v = (gamma-1/gamma)/v
# gamma = 1+E/m0c2
# as v->0, gamma->1, Tb/Td ~ E/m0c2/v ~ v ~ sqrt(E), Td/Tb ~ 1/sqrt(E)
# as v->c, gamma->inf, Tb/Td ~ gamma ~ E, Tb/Td ~ E, Td/Tb ~ 1/E

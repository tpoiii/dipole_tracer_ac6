"""
plot of AC6 dos1 response along with exponential weightings
"""

import numpy as np
import matplotlib.pyplot as plt
from dipole_tracer import read_ac6_iso_response_csv,get_energy_limits
from scipy.integrate import cumtrapz

iso = read_ac6_iso_response_csv('data/ac6a_response_iso.csv')
dos1resp = {'keV':iso['MeV']*1e3,'DOS1':iso['DOS1']} # save for later

plt.close('all')

E0s = [20,70,150] # energies to look at, keV
colors = ['r','g','b']

# make figure showing response weighted by exp(-E/E0)
plt.figure()
plt.loglog(dos1resp['keV'],dos1resp['DOS1'],'k-',lw=2,label='DOS1 Response, cm$^2$sr')
Rmax = dos1resp['DOS1'][dos1resp['keV']<300].max()
for (i,E0) in enumerate(E0s):
    elimits = get_energy_limits(dos1resp,E0)
    plt.loglog(elimits['Epeak'],elimits['Rjpeak']/elimits['Rjpeak']*Rmax,'^',color=colors[i])
    plt.loglog(elimits['Emid'],elimits['Rjmid']/elimits['Rjpeak']*Rmax,'o',color=colors[i])
    plt.loglog(dos1resp['keV'],elimits['Rj']/elimits['Rjpeak']*Rmax,'-',label='E$_s$=%g keV, max @ %.0f keV, 50%%$\int$ @ %.0f keV' % (E0,elimits['Epeak'],elimits['Emid']),color=colors[i])
    
plt.xlabel('Electron Energy, keV')
plt.ylabel('cm$^2$sr')
plt.ylim([1e-5,1e-3])
plt.legend()
plt.grid('on')
plt.title('AC6 DOS1 Response, Weighted by e$^{-E/E_s}$')
plt.savefig('figures/dos1-response.png')

# make figure showing cumulative response weighted by exp(-E/E0)
plt.figure()
for (i,E0) in enumerate(E0s):
    y = dos1resp['DOS1']*np.exp(-dos1resp['keV']/E0) # weighted response
    c = cumtrapz(y,dos1resp['keV'],initial=0) # cumulative weighted response
    c = 100*c/c[-1] # normalize to percent
    imid = np.argmin(np.abs(c-50)) # find midpoint
    Emid = dos1resp['keV'][imid]
    plt.semilogx(Emid,50,'o',color=colors[i])
    plt.semilogx(dos1resp['keV'],c,'-',label='E$_s$=%g keV, 50%%$\int$ @ %.0f keV' % (E0,Emid),color=colors[i])
    
plt.xlabel('Electron Energy, keV')
plt.ylabel('Weighted Cumulative Response, %')
plt.ylim([0,100])
plt.legend()
plt.grid('on')
plt.title('AC6 DOS1 Cumulative Response, Weighted by e$^{-E/E_s}$')
plt.savefig('figures/dos1-cumulative-response.png')

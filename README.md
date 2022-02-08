# dipole_tracer_ac6
Dipole Tracer and AeroCube-6 code for analysis of precipitation curtains

This software is provided under the terms of the MIT license provided in LICENSE.txt.

Initial developer: Paul O'Brien (paul.obrien@aero.org)

The following python files are included. Documentation is provided via docstrings
- dipole_tracer.py - the main library that includes the dipole tracer, source regions, and associated utilities
- bounces-per-drift.py - code to make figures describing bounce and drift timescales
- load_ac6_data_np.py - reader for AC6 data (courtesy of Mike Shumko)
- response-fig.py - figure showing the AC6 sensor response
- scene1.py - figures showing a single microburst source at different drift distances before AC6
- scene2.py - figures showing a multiple microburst sources at different drift distances before AC6
- scene-E0.py - figures showing the dependence of curtain formation on the energy spectrum
- scene-real.py - figures reconstructing a real scene observed by AC6 with equatorial microburst sources
- odc_util.py - utlities for computing adiabatic invariants and associated parameters
- tictoc.py - utility to time operations

imported standard/common python modules:
numpy, matplotlib, os, sys, warnings, json, multiprocessing, scipy, pickle, datetime, dateutil

non-standard dependencies:
spacepy - open source library of space science calculations. Used for reading NASA CDF formatted data

data included in repository:

AC6 CSV files for 2015 Feb 07
- AC6-A_20150207_L2_10Hz_V03.csv
- AC6-A_20150207_L2_att_V03.csv
- AC6-A_20150207_L2_coords_V03.csv
- AC6-A_20150207_L2_survey_V03.csv
- AC6-B_20150207_L2_10Hz_V03.csv
- AC6-B_20150207_L2_att_V03.csv
- AC6-B_20150207_L2_coords_V03.csv
- AC6-B_20150207_L2_survey_V03.csv

AC6 response CSV file (also available from Zenodo doi:10.5281/zenodo.5796340)
- ac6a_response_iso.csv
- ac6b_response_iso.csv

RBSP-ECT combined spectrum CDFs for 2015 Feb 07 (available from rbsp-ect.newmexicoconsortium.org)
- rbspa_ect-elec-L2_20150207_v2.1.0.cdf
- rbspb_ect-elec-L2_20150207_v2.1.0.cdf

EMFSIS plasmapause crossings (available from emfisis.physics.uiowa.edu)
- rbspa_inner_plasmapause_list2.dat.txt
- rbspb_inner_plasmapause_list2.dat.txt

AC6 data are in cdaweb.gsfc.nasa.gov (CDFs). CSV files can be obtained at:
  https://spdf.gsfc.nasa.gov/pub/data/aaa_smallsats_cubesats/aerocube/aerocube-6/ and https://rbspgway.jhuapl.edu/ac6



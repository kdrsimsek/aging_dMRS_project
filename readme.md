# Age-Trajectories of Higher-Order Diffusion Properties of Major Brain Metabolites in Cerebral and Cerebellar Gray Matter Using In Vivo Diffusion-Weighted MR Spectroscopy at 3T

Quantified metabolite areas using LCModel can be found in "data/dMRS_aging_fitResults.xlsx, including participants' demographic data".
Additionally, b-values for all directions can be found in "data/b_values_from_chrono.xlsx".

# Diffusion Analysis
The scripts for diffusion analysis can be found in "scripts/dMRS_aging_analysis.py". The script fits data to all diffusion models and signal representations and exports the results as JSON file in data/dMRS_aging_diffAnalysis_least_squares.json
The script file in scritps/dMRS_aging_figures.py generates the figures in the paper using the exported results file.

The scripts require some python libraries: pandas, numpy, lmfit, matplotlib, scipy, json, asyncio, joblib and statsmodels

# License
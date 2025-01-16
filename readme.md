# Age-Trajectories of Higher-Order Diffusion Properties of Major Brain Metabolites in Cerebral and Cerebellar Gray Matter Using In Vivo Diffusion-Weighted MR Spectroscopy at 3T

Quantified metabolite areas using LCModel can be found in "data/dMRS_aging_fitResults.xlsx, including participants' demographic data".
Additionally, b-values for all directions can be found in "data/b_values_from_chrono.xlsx".

# Diffusion Analysis
The scripts for diffusion analysis can be found in "scripts/dMRS_aging_analysis.py". The script fits data to all diffusion models and signal representations and exports the results as JSON file in "data/dMRS_aging_diffAnalysis_least_squares.json".
The script file in "scritps/dMRS_aging_figures.py" generates the figures in the paper using the exported results file.

The scripts require some python libraries: pandas, numpy, lmfit, matplotlib, scipy, json, asyncio, joblib and statsmodels

# Acknowleedgements
•	The authors would like to thank Dr. Edward J. Auerbach and Dr. Małgorzata Marjańska for providing us with the dMRS sequence for the Siemens platform and simulating the basis set for spectral fitting.
•	

# Funding
•	This work, Kadir Şimşek and Marco Palombo are supported by UKRI Future Leaders Fellowship (MR/T020296/2).
•	FB, CG and SL acknowledge support from the programs 'Institut des neurosciences translationnelle' ANR-10-IAIHU-06 and 'Infrastructure d'avenir en Biologie Santé' ANR-11-INBS-0006. 

# Citation
Şimşek, K., Gallea, C., Genovese, G., Lehéricy, S., Branzoli, F. and Palombo, M. (2025), Age-Trajectories of Higher-Order Diffusion Properties of Major Brain Metabolites in Cerebral and Cerebellar Gray Matter Using In Vivo Diffusion-Weighted MR Spectroscopy at 3T. Aging Cell e14477. https://doi.org/10.1111/acel.14477

# License
This article is published under an Open Access license and is freely available to the public. The content is licensed under the terms of the Creative Commons Attribution 4.0 International License (CC BY 4.0).
For further details about the license, please visit <a href="https://creativecommons.org/" target="_blank"> the Creative Commons website </a>.

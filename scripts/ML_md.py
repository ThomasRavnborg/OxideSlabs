from src.activeNEP import ActiveLearningNEP

NEP = ActiveLearningNEP('results/ALnep')

#NEP.run_MD('npt_ber_production')
NEP.run_MD('nve_production')
NEP.run_MD('pimd_8_production')
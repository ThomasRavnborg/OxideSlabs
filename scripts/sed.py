from src.activeNEP import ActiveLearningNEP

NEP = ActiveLearningNEP('results/ALnep')

NEP.calculate_sed(path='nve_production/Ba8O24Ti8/500K')


"""
for frame_step in [1, 2, 5, 10, 20]:
    NEP.calculate_sed(path='nve_production_test2/Ba8O24Ti8/500K', frame_step=frame_step)

for i in range(5):
    frame_start = i * 2000
    frame_stop = (i + 1) * 2000 - 1
    NEP.calculate_sed(path='nve_production_test2/Ba8O24Ti8/500K',
                      frame_start=frame_start, frame_stop=frame_stop)
"""
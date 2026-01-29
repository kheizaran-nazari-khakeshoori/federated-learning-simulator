from algorithms import get_algorithm
print("benchmark fedavg vs fedprox on alpha 0.1/10")

# pareto curve comm vs acc
import matplotlib.pyplot as plt
def pareto(xs, ys): plt.plot(xs, ys)

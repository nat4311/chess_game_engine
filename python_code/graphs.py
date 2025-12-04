import matplotlib.pyplot as plt

# initial python timings (ram overflow)
y1 = [ 3.437511622905731e-05, 0.009956791065633297, 0.022661556489765644, 1.4706027386710048, 1.5414285976439714, ]
# still python, after removing children cache from GameStateNode
y2 = [ 0.0006825476884841919, 0.007834275253117085, 0.017839105799794197, 0.764815291389823, 0.8085004044696689, 162.43249557446688, 69.25481910258532, ]
# python, after moving legal_move_generation to c++
y3 = [ 8.558016270399094e-05, 0.0007701413705945015, 0.0020072944462299347, 0.07913886103779078, 0.08957733120769262, 15.086321806535125, 7.631460613571107, ]
# first c++ GameStateNode (no move ordering)
y4 = [ 1.7e-05, 0.000131, 0.000345, 0.019428, 0.022782, 4.34886, 1.94325, 433.817, ]
# c++, move ordering
y5 = [ 1.9e-05, 7.6e-05, 0.00112, 0.006624, 0.072729, 0.582312, 4.93559, 35.927, 191.175, 1529.02, ]
# c++, open mp
y6 = [ 1.9e-05, 8.4e-05, 0.000601, 0.002885, 0.021282, 0.10138, 1.39301, 7.34563, 70.1452, 409.803, 2877.88, ]


####################### PLOT 4
plt.close()
plt.figure(figsize=(12,5))
plt.title("Open MP Parallelization")
plt.ylabel("Time (s)")
plt.xlabel("Search Depth")
plt.semilogy([i+1 for i in range(len(y5))], y5, label="single thread")
plt.semilogy([i+1 for i in range(len(y6))], y6, label="parallelized")
plt.legend()
plt.grid()
plt.savefig("fig4.png")

####################### PLOT 3
plt.close()
plt.figure(figsize=(12,5))
plt.title("Improved Algorithm")
plt.ylabel("Time (s)")
plt.xlabel("Search Depth")
plt.semilogy([i+1 for i in range(len(y4))], y4, label="before move ordering")
plt.semilogy([i+1 for i in range(len(y5))], y5, label="after move ordering")
plt.legend()
plt.grid()
plt.savefig("fig3.png")

####################### PLOT 2
plt.close()
plt.figure(figsize=(12,5))
plt.title("Migrated Child Node Generation to c++")
plt.ylabel("Time (s)")
plt.xlabel("Search Depth")
plt.semilogy([i+1 for i in range(len(y2))], y2, label="child node generation in python")
plt.semilogy([i+1 for i in range(len(y3))], y3, label="child node generation in c++")
plt.semilogy([i+1 for i in range(len(y4))], y4, label="everything in c++")
plt.legend()
plt.grid()
plt.savefig("fig2.png")

####################### PLOT 1
plt.close()
plt.figure(figsize=(12,5))
plt.title("Effect of Removing Children Cache")
plt.ylabel("Time (s)")
plt.xlabel("Search Depth")
plt.semilogy([i+1 for i in range(len(y1))], y1, label="Initial GameStateNode")
plt.semilogy([i+1 for i in range(len(y2))], y2, label="After removing children cache")
plt.legend()
plt.grid()
plt.savefig("fig1.png")

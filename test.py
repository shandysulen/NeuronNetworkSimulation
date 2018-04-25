import numpy as np
import matplotlib.pyplot as plt

def getPSP(tau, alpha, Q, t):
    return (Q/(alpha * np.sqrt(t))) * (np.exp(-1 * np.power(alpha, 2) / t)) * (np.exp(-1 * t / tau))

def getAHP(t):
    return (-1000 * np.exp(-1 * t / 1.2))

tau = 20
Q = 5
alpha = 2
time_range = 100
interval = 0.1
t = np.arange(interval, time_range, interval)
cutoff = 50
print(cutoff)

plt.xlabel('Time (ms)')
plt.ylabel('Potential Voltage (mV)')
plt.title('Neuronal Spike')
PSP = getPSP(tau, alpha, Q, t)
AHP = getAHP(t)
print("Len of PSP:",len(PSP))
print("Len of AHP:",len(AHP))
AHP = np.concatenate([[0.0] * int(cutoff),AHP[:int(len(AHP) - cutoff)]])
print(PSP+AHP)
print("Max:", max(PSP+AHP))
plt.plot(t, PSP + AHP, 'r')
plt.show()

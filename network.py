import random
import numpy as np
import matplotlib.pyplot as plt
import time

# TIME GLOBALS
time_range = 1000
interval = 0.1
numIntervals = int(time_range / interval)
cutoff_time = 10
t = np.arange(0.1, time_range, interval)
voltage = [0] * len(t)


class Neuron:
    ID = 0
    def __init__(self, type):
        self.type = type
        self.tau = 20
        self.id = self.__class__.ID
        self.__class__.ID += 1
        self.spikeTimes = []
        self.PSP = [0] * len(t)

        # Set Excitatory or Inhibitory parameters
        if self.type == "Excitatory":
            self.Q = random.uniform(5,11)
            self.alpha = random.uniform(1,3)
            self.threshold = 5
        else:
            self.Q = random.uniform(30,61)
            self.alpha = random.uniform(1,1.1)
            self.threshold = 10

        if self.id % 4 == 0:
            self.special = True
        else:
            self.special = False

    def getPSP(self,tau, alpha, Q, t):
        return (Q/(alpha * np.sqrt(t))) * (np.exp(-1 * np.power(alpha, 2) / t)) * (np.exp(-1 * t / tau))

    def getAHP(self,t):
        return (-1 * np.exp(-1 * t / 1.2))

    def generateSpike(self, time_step):
        spike_range = np.arange(0.1, 100, 0.1)
        cutoff = 50

        PHP = self.getPSP(self.tau, self.alpha, self.Q, spike_range)
        AHP = self.getAHP(spike_range)
        spike = np.concatenate([[0.0] * cutoff,AHP[:int(len(AHP) - cutoff)]])
        trailing_zeros = [0] * (len(t) - len(spike) - time_step)
        newPSP = np.concatenate([self.PSP[0:time_step], spike, trailing_zeros])
        self.PSP = newPSP
        self.PSP = self.PSP[0:numIntervals-1]

class NeuronList:
    def __init__(self, numNeurons):
        numExc = int(0.8 * numNeurons)
        numInh = int(0.2 * numNeurons)
        self.list = []
        for i in range(numExc):
            self.list.append(Neuron("Excitatory"))
        for i in range(numInh):
            self.list.append(Neuron("Inhibitory"))

    def printList(self):
        for i in range(len(self.list)):
            print("Neuron #" + str(self.list[i].id) + ": " + self.list[i].type + " | Q=" + str(self.list[i].Q) + " | alpha=" + str(self.list[i].alpha))

class PoissonNeuron:
    """
    Models a single neuron to be used in a larger network

    Fields:
    numPoissonNeurons = number of input neurons to receive signals from
    threshold = voltage potential (mV) minimum that must be reached in order to
    generate spikes
    """
    ID = 0
    def __init__(self):
        self.threshold = 5
        self.type = "Excitatory"
        self.Q = random.uniform(5,11)
        self.tau = 20
        self.alpha = random.uniform(1,3)
        self.PSP = [0] * len(t)
        self.spikeTimes = []
        self.id = self.__class__.ID
        self.__class__.ID += 1

    def generateSpike(self):
        return list((self.Q/(self.alpha * np.sqrt(t))) * (np.exp(-1 * np.power(self.alpha, 2) / t)) * (np.exp(-1 * t / self.tau)))

class PoissonNeuronList:
    def __init__(self, numInputNeurons):
        self.list = []
        for i in range(numInputNeurons):
            self.list.append(PoissonNeuron())

    def printList(self):
        for i in range(len(self.list)):
            print("Neuron #" + str(self.list[i].id) + ": " + self.list[i].type + " | Q=" + str(self.list[i].Q) + " | alpha=" + str(self.list[i].alpha) + " | PSP=" + str(self.list[i].PSP))

    def generateSpikeTimes(self, lam):
        print("Generating spike times for input Poisson Neurons...")
        for i in range(len(self.list)):
            for j in range(len(t)):
                if random.random() < lam:
                    self.list[i].spikeTimes.append(j)

    def calculatePSPs(self):
        print("Calculating PSP's for input Poisson Neurons...")
        for i in range(len(self.list)):
            for j in range(len(self.list[i].spikeTimes)):
                self.list[i].PSP = self.list[i].PSP[0:self.list[i].spikeTimes[j]] + self.list[i].generateSpike()
                self.list[i].PSP = self.list[i].PSP[0:numIntervals-1]

    def printSpikeTimes(self):
        print("Printing Poisson Neuron spike times...")
        for i in range(len(self.list)):
            print("Neuron #",self.list[i].id,": ",self.list[i].spikeTimes)

class SynapseMatrix:
    def __init__(self, neuronList, numSynapses, neuronsToAdd):
        self.rows = len(neuronList.list)
        self.cols = numSynapses
        self.matrix = [[0 for x in range(self.cols + neuronsToAdd)] for y in range(self.rows)]

    def populateMatrix(self):
        for i in range(self.rows):
            selected = []
            for j in range(self.cols):
                while True:
                    neuronIndex = random.randrange(0,self.rows)
                    if neuronIndex not in selected:
                        self.matrix[i][j] = neuronList.list[neuronIndex]
                        selected.append(neuronIndex)
                        break

    def addPoissonNeurons(self, pNeuronList, neuronsToAdd):
        for i in range(self.rows):
            selected = []
            for j in range(neuronsToAdd):
                while True:
                    neuronIndex = random.randrange(0,len(pNeuronList.list))
                    if neuronIndex not in selected:
                        self.matrix[i][self.cols + j] = pNeuronList.list[neuronIndex]
                        selected.append(neuronIndex)
                        break

    def printMatrix(self, neuronsToAdd):
        extra_cols = neuronsToAdd
        for i in range(self.rows):
            print("----------------------------")
            for j in range(self.cols + extra_cols):
                print("Row:", i, "Col:", j, "|", "Neuron Num #", self.matrix[i][j].id, "|", self.matrix[i][j].type, "| Q=", self.matrix[i][j].Q)

    def printPSPatTimeStep(self, time_step, row, col):
        print("Row: " + str(row) + ", Col: " + str(col) + " | PSP: " + str(self.matrix[row][col].PSP[time_step]))

if __name__ == "__main__":
    start_time = time.time()

    # Create and print a single list of Neurons that shall be referenced from
    numNeurons = 100
    neuronList = NeuronList(numNeurons)

    # Create adjacency list for synapses of each neuron
    numSynapses = 10
    neuronsToAdd = 10
    numInputNeurons = 100
    sMatrix = SynapseMatrix(neuronList, numSynapses, neuronsToAdd)
    sMatrix.populateMatrix()

    # Create and print Poisson Neuron List
    pNeuronList = PoissonNeuronList(numInputNeurons)

    # Add a selected number of neurons to add to the Poisson Neuron List
    sMatrix.addPoissonNeurons(pNeuronList, neuronsToAdd)

    # Set lambda for Poisson Neurons
    lam = 0.1 # should be 0.0001

    # Generate spike times of input Poisson Neurons
    pNeuronList.generateSpikeTimes(lam)

    # Calculate PSP's for all input Poisson Neurons
    pNeuronList.calculatePSPs()

    # Initialize Network Spikes
    networkSpikes = [0] * (int(100 / interval) - 1)
    t_spikes = np.arange(0.1, 100, interval)

    print("Generating spikes throughout the whole network...")
    # Update PSP's for all neurons
    for time_step in range(len(t)):
        totalPSP = 0
        for row in range(sMatrix.rows):
            sumPSP = 0
            for col in range(sMatrix.cols + neuronsToAdd):
                sumPSP += sMatrix.matrix[row][col].PSP[time_step]
                # sMatrix.printPSPatTimeStep(time_step, row, col)
            # print("Total PSP for Row", row, ":", sumPSP)
            if neuronList.list[row].special:
                totalPSP += sumPSP
            if time_step > (len(t) - int((100/interval))) and sumPSP > neuronList.list[row].threshold:
                logging_point = len(t) - int(100/interval) + 1                
                networkSpikes[time_step - logging_point] += 1
                neuronList.list[row].generateSpike(time_step) # we want change to appear at the next iteration
        # Log totalPSP for 250 neurons for that time_step
        voltage[time_step] = totalPSP

    end_time = time.time()
    print("Time: " + str(round(end_time - start_time, 2)) + " seconds")


    # Plotting total number of spikes over last 100 msec
    plt.figure(1)
    plt.xlabel('Time (ms)')
    plt.ylabel('Number of Spikes')
    plt.title('Total Spikes')
    plt.gca().invert_xaxis()
    plt.plot(t_spikes, networkSpikes, 'r')
    plt.show()

    # Plotting potential voltage for 250 neurons
    plt.figure(2)
    plt.xlabel('Time (ms)')
    plt.ylabel('Potential Voltage (mV)')
    plt.title('250 Neuron Total Potential')
    plt.gca().invert_xaxis()
    plt.plot(t, voltage, 'r')
    plt.show()

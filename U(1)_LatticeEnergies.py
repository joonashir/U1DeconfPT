import numpy as np

import matplotlib.pyplot as pl

from math import sqrt
from math import e
from math import pi

from numba import jit


#-------------------------------------------------------------------------------
#       Note for using autocorrelations

    #For low values of beta, the cut-off point is very low.
    #This causes the code to produce too few values for the autocorrelation function.

#-------------------------------------------------------------------------------
#       latt

def initlatt(sizeT, sizeX, randInit):   #Initializes the lattice with a boolean for random (hot, True) or all zeros (cold, False) initialization
    if (randInit):
        return np.array([[[2*pi*np.random.random() for d in range(2)] for x in range(sizeX)] for t in range(sizeT)])
    else:
        return np.array([[[0.0 for d in range(2)] for x in range(sizeX)] for t in range(sizeT)])


#-------------------------------------------------------------------------------
#       Staples, plaquette and action
@jit
def localAction(latt, t, x, sizeT, sizeX, dir):     #Produces the part of the action that is dependent on the specified link
    if (dir==0):
        return -(plaquette(latt, t, x, sizeT, sizeX) + plaquette(latt, t, x-1, sizeT, sizeX))
    else:
        return -(plaquette(latt, t, x, sizeT, sizeX) + plaquette(latt, t-1, x, sizeT, sizeX))

@jit
def plaquette(latt, t, x, sizeT, sizeX):                    #Can be evaluated only in one direction
    return  e**(1j * (latt[t][x][1] + latt[t][(x+1)%sizeX][0] - latt[(t+1)%sizeT][x][1] - latt[t][x][0]) )


def pseudoEnergy(latt, sizeT, sizeX):           #Calculates the lattice sstem equivalent to energy
    E = 0
    for t in range(sizeT):
        for x in range(sizeX):
            E -= plaquette(latt, t, x, sizeT, sizeX).real

    return 1 + E/(sizeT*sizeX)

def action(latt, beta, sizeT, sizeX):
    S = 0
    for t in range(sizeT):
        for x in range(sizeX):
            S += 1-plaquette(latt, t, x, sizeT, sizeX).real

    return beta*S


def plaquetteAvg(latt, sizeT, sizeX):
    pS = 0
    for t in range(sizeT):
        for x in range(sizeX):
            pS += plaquette(latt, t, x, sizeT, sizeX)
    return pS/(sizeT * sizeX)

#-------------------------------------------------------------------------------
#       Evolution of the latt

@jit
def stepForLink(latt, beta, t, x, sizeT, sizeX, dir, width):    #Suggest a step for the specified link and accepts it with the correct probability
    local = localAction(latt, t, x, sizeT, sizeX, dir).real

    delta = width * (np.random.random()-0.5)

    latt[t][x][dir] += delta

    DeltaS = beta*(localAction(latt, t, x, sizeT, sizeX, dir).real - local)

    p = np.random.random()

    if (e**(-DeltaS)<p):
        latt[t][x][dir] -= delta
    else:
        if (latt[t][x][dir] >= pi):
            latt[t][x][dir] -= 2*pi
        elif (latt[t][x][dir] <= -pi):
            latt[t][x][dir] += 2*pi

@jit
def sweep(latt, beta, sizeT, sizeX, width):         #Does the step function for each link
    gates = np.array([[0,0],[0,1],[1,0],[1,1]])     #For detailed balance:
    np.random.shuffle(gates)                        #   Only decoupled groups are done together
                                                    #   Order for sweeping over the groups is radomized
    for g in gates:
        for t in range(sizeT):
            for x in range(sizeX):
                if (g[0]  == 0):
                    if (x % 2 == g[1]):
                        stepForLink(latt, beta, t, x, sizeT, sizeX,  g[0], width)
                else:
                    if (t % 2 == g[1]):
                        stepForLink(latt, beta, t, x, sizeT, sizeX,  g[0], width)


def sweeps(latt, beta, sizeT, sizeX, width, number_of_steps_per_measurement):   #Specified number of sweeps
    for j in range(number_of_steps_per_measurement):
        sweep(latt, beta, sizeT, sizeX, width)

#-------------------------------------
#Test algorithms


def acceptanceRateTest(latt, sizeT, sizeX, beta, width, equilSteps, stepNumb):  #Tests the acceptance rate
    for j in range(number_of_steps_per_measurement):
        sweep(latt, beta, sizeT, sizeX, width)

    rates = []
    for i in range(stepNumb):
         rates.append(sweepForAcceptanceRateTest(latt, beta, sizeT, sizeX, width))

    print("Acceptance rate: ", np.mean(rates), "+/-", np.std(rates)/sqrt(len(rates)-1))

@jit
def sweepForAcceptanceRateTest(latt, beta, sizeT, sizeX, width):
    gates = np.array([[0,0],[0,1],[1,0],[1,1]])
    np.random.shuffle(gates)

    succs = 0

    for g in gates:
        for t in range(sizeT):
            for x in range(sizeX):
                if (g[0]  == 0):
                    if (x % 2 == g[1]):
                        succs += stepForacceptanceRateTest(latt, beta, t, x, sizeT, sizeX, g[0], width)
                else:
                    if (t % 2 == g[1]):
                        succs += stepForacceptanceRateTest(latt, beta, t, x, sizeT, sizeX, g[0], width)

    return succs / (2*sizeT*sizeX)

@jit
def stepForacceptanceRateTest(latt, beta, t, x, sizeT, sizeX, dir, width):
    local = localAction(latt, t, x, sizeT, sizeX, dir).real

    delta = width * (np.random.random()-0.5)

    latt[t][x][dir] += delta

    DeltaS = beta*(localAction(latt, t, x, sizeT, sizeX, dir).real - local)

    p = np.random.random()

    if (e**(-DeltaS)<p):
        latt[t][x][dir] -= delta
        return 0
    else:
        if (latt[t][x][dir] >= pi):
            latt[t][x][dir] -= 2*pi
        elif (latt[t][x][dir] <= -pi):
            latt[t][x][dir] += 2*pi
        return 1



#-------------------------------------------------------------------------------
#   Autocorrelation



def productTForAutocorrs(data, t):  #A subroutine for the function autocorrelations
    if (t == len(data)-1):
        return [0,0]

    prodList = []
    for i in range(len(data)-t):
        prodList.append(data[i]*data[i+t])

    prod = np.mean(prodList)
    err = np.std(prodList)/sqrt(len(prodList)-1)

    return [prod, err]


def autocorrelations(data):     #Computes the autocorrelation function for the input data
    avg = np.mean(data)
    avgErr = np.std(data)/sqrt(len(data)-1)

    autoCorrels = []
    for t in range(len(data)):
        autoCorrels.append(productTForAutocorrs(data, t))
        autoCorrels[-1][0] -= avg**2
        autoCorrels[-1][1] = sqrt(autoCorrels[-1][1]**2 + (2*avg*avgErr)**2 )
        if (autoCorrels[-1][0] <= autoCorrels[-1][1]):                  #NOTE!  For lower betas this gives too few values for autocorrelation function!
            break
        #if (t>5):                                                      #This should give the wanted amount of values
        #    break

    autoCorrels = np.transpose(autoCorrels).tolist()

    print("\n\nAutocorrelations:")
    print(autoCorrels[0])
    print("\nErrors of autocorrelations:")
    print(autoCorrels[1])

    return autoCorrels





#-------------------------------------------------------------------------------
#       A Run

#A run that gives the lattice system equivalent of the energy and the autocorrelation function for the energy
def runPseudoEnergy(latt, beta, sizeT, sizeX, width, equilSteps, number_of_steps_per_measurement, number_of_measurements):
    sweeps(latt, beta, sizeT, sizeX, width, equilSteps)

    psEnergs = []
    for i in range(number_of_measurements):
        sweeps(latt, beta, sizeT, sizeX, width, number_of_steps_per_measurement)
        psEnergs.append(pseudoEnergy(latt, sizeT, sizeX))

    avgPsEn = np.mean(psEnergs)
    errPsEn = np.std(psEnergs)/sqrt(number_of_measurements-1)


    print("\n\n\n\nBeta: ", beta, ", Random init: ", randInit, ", Width: ", width, ", Latt size: ", sizeT, "x", sizeX)
    print("\nPseudo energy: ", avgPsEn, "+/-", errPsEn)

    return [avgPsEn, errPsEn], autocorrelations(psEnergs)

#-------------------------------------------------------------------------------
#       Variables

number_of_measurements = 500000
number_of_steps_per_measurement = 1

equilSteps = 1000


#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#       Runs






            #Finding the pseudo energies



"""
widths = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) * 2*pi    #Autocorrelatoin function needs to be modified to give more values
betas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

widths = np.array([1.0, 0.95, 0.86, 0.79, 0.74, 0.69, 0.65, 0.62, 0.59, 0.57]) * 2*pi    #Autocorrelatoin function needs to be modified to give more values
betas = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

widths = np.array(  [0.545, 0.525,  0.51,   0.49,   0.48,   0.465,  0.455,  0.44,   0.43,   0.42]) * 2*pi
betas =             [2.1,   2.2,    2.3,    2.4,    2.5,    2.6,    2.7,    2.8,    2.9,    3.0]
"""
widths = np.array(  [0.415,  0.405,  0.395,  0.39,   0.385,  0.375,  0.37,   0.365,  0.36,   0.355]) * 2*pi
betas =             [3.1,    3.2,    3.3,    3.4,    3.5,    3.6,    3.7,    3.8,    3.9,    4.0]



pseudoEnergies = []
theirAutocorrelations = []
for i in range(10):
    width = widths[i]
    beta = betas[i]
    for size in [8, 12, 16, 20]:
        sizeT = size
        sizeX = size
        for randInit in [True, False]:
            latt = initlatt(sizeT, sizeX, randInit)
            psE, thAc = runPseudoEnergy(latt, beta, sizeT, sizeX, width, equilSteps, number_of_steps_per_measurement, number_of_measurements)
            pseudoEnergies.append(psE); theirAutocorrelations.append(thAc)

print("\n\n\n\nPseudo energies: ", pseudoEnergies)
print("\n\n\n\nAutocorrelations: ", theirAutocorrelations)

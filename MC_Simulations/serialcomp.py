
import numpy as np
from scipy.stats import norm
import pandas as pd
from joblib import Parallel, delayed
import time

#Closed formula
def Bac(t, x, k, sigma, T):
    dbac = (x-k)/(sigma*np.sqrt(T-t))
    norm_cumm = norm.cdf(dbac)
    norm_pdf =  norm.pdf(dbac)
    return (x-k)*norm_cumm + norm_pdf*sigma*np.sqrt(T-t)

#Sto Volatilities model defs
def Heston_Volatility2(n, m, T, brownB, sigma0, params):
    dt = T / m
    kappa = params['kappa']
    theta = params['theta']
    vega = params['vega']
    
    vols = np.zeros((n, m+1))
    volsP = np.zeros((n, m+1))
    for i in range(n):
        vols[i, 0] = sigma0**2
        volsP[i, 0] = sigma0**2

    for i in range(m):
        for j in range(n):
            dW = brownB[j, i] * np.sqrt(dt)
            vols[j, i+1] = vols[j, i] + kappa * (theta - volsP[j, i]) * dt + vega * np.sqrt(volsP[j, i]) * dW
            volsP[j, i+1] = np.maximum(vols[j, i+1], 0) #0 per negatives
        
    return np.sqrt(volsP)


#Conditional Monte-Carlo
def CondMC(S0, K, sigma0, n, m, T, volupdate=None, volupdateparams=None,rho=0):
    brownB=np.random.normal(0,1,(n,m+1))
    dt = T/m
    int_dw=np.zeros(n)
    Bacs = np.zeros(n)

    sigma1=volupdate(n, m, T, brownB, sigma0, volupdateparams)

    #v0 = np.sum(((sigma1[:, :]))*np.sqrt(dt), axis=1)
    v0 = np.sqrt(np.sum(((sigma1[:, :])**2) * dt, axis=1))/ T 

    for j in range(0,m):
        for i in range(n):
        #int_dw=int_dw+sigma1[:, j]*(brownB[:,j+1] - brownB[:, j])*np.sqrt(T/m)
            int_dw=int_dw+sigma1[i, j]*(brownB[i,j+1] - brownB[i, j])*np.sqrt(dt)  

    for i in range(n):
        S0_prime = S0 + rho*(int_dw[i])
        Bacs[i] = Bac(0, S0_prime, K, np.sqrt(1 - rho**2)*v0[i], T)

    return np.asarray(Bacs)

def rawMC(S0, K, sigma0, n, m, T, volupdate=None, volupdateparams=None,rho=0):
    brownB=np.random.normal(0,1,(n,m+1))
    brownW=np.random.normal(0,1,(n,m+1))
    dt = np.sqrt(T/m)
    assetpaths = np.zeros((n,m+1))
    for i in range(n):
        assetpaths[i,0] = S0
    sigma1=volupdate(n, m, T, brownB, sigma0, volupdateparams)
    for i in range(m):
        for j in range(n):
            assetpaths[j, i+1] = assetpaths[j,i]+sigma1[j,i]*(rho*brownW[j,i]*dt + np.sqrt(1-(rho**2))*brownB[j,i]*dt)

    return np.asarray(assetpaths[:,m])-K

print("starting raw")
rawstart = time.time()
print(1)
payoffraw = rawMC(700, 698.5, 0.43, 1500, 2500, 5, volupdate=Heston_Volatility2, volupdateparams={'kappa': 5, 'theta': 0.46, 'vega': 0.6}, rho=0.3)
for i in range(11):
    print(i+2)
    payoffraw_it = rawMC(700, 698.5, 0.43, 1500, 1800, 5, volupdate=Heston_Volatility2, volupdateparams={'kappa': 5, 'theta': 0.46, 'vega': 0.6}, rho=0.3)
    payoffraw = np.concatenate((payoffraw, payoffraw_it))
rawend = time.time()
rawtime = rawend - rawstart

condstart = time.time()
print(1)
payoffcond = CondMC(700, 698.5, 0.43, 1500, 2500, 5, volupdate=Heston_Volatility2, volupdateparams={'kappa': 5, 'theta': 0.46, 'vega': 0.6}, rho=0.3)
print("starting Cond")
for i in range(11):
    print(i+2)
    payoffcond_it = CondMC(700, 698.5, 0.43, 1500, 1800, 5, volupdate=Heston_Volatility2, volupdateparams={'kappa': 5, 'theta': 0.46, 'vega': 0.6}, rho=0.3)
    payoffcond = np.concatenate((payoffcond, payoffcond_it))
condend = time.time()
condtime = condend-condstart

pdmethods = {'Method':["Serial Raw Monte Carlo", "Serial Conditional Monte Carlo"], 'Average Price':[np.mean(payoffraw), np.mean(payoffcond)], 'Prices Standard Deviation':[np.var(payoffraw), np.var(payoffcond)], 'Exectution Time':[rawtime, condtime]}
pdmethods = pd.DataFrame(pdmethods)
print(payoffraw)
print('--->Raw: ',np.mean(payoffraw), 'std:', np.var(payoffraw))
print('--->Cond:',np.mean(payoffcond), 'std:', np.var(payoffcond))
pdmethods.to_csv('prixSimsResults/serialmethods.csv')


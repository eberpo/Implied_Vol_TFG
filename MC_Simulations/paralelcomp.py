
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
    vols[:, 0] = sigma0**2
    volsP = np.zeros((n, m+1))
    volsP[:, 0] = sigma0**2

    for i in range(m):
        dW = brownB[:, i] * np.sqrt(dt)
        vols[:, i+1] = vols[:, i] + kappa * (theta - volsP[:, i]) * dt + vega * np.sqrt(volsP[:, i]) * dW
        volsP[:, i+1] = np.maximum(vols[:, i+1], 0) #0 per negatives
    
    return np.sqrt(volsP)

def Lognormal_Volatility(n, m, T, brownB, sigma0, params):
    dt = T / m
    kappa = params['kappa']
    theta = params['theta']
    vega = params['vega']
    
    vols = np.zeros((n, m+1))
    log_vols = np.zeros((n, m+1))
    
    log_vols[:, 0] = np.log(sigma0)
    
    for i in range(m):
        dW = brownB[:, i] * np.sqrt(dt)
        log_vols[:, i+1] = log_vols[:, i] + kappa * (np.log(theta) - log_vols[:, i]) * dt + vega * dW
    
    vols = np.exp(log_vols)  # Convert back to volatility scale
    
    return vols

def Static_Volatility(n, m, T, brownB, sigma0, params):
    return np.full((n, m+1), sigma0)

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
        #int_dw=int_dw+sigma1[:, j]*(brownB[:,j+1] - brownB[:, j])*np.sqrt(T/m)
        int_dw=int_dw+sigma1[:, j]*(brownB[:,j+1] - brownB[:, j])*np.sqrt(dt)  

    for i in range(n):
        S0_prime = S0 + rho*(int_dw[i])
        Bacs[i] = Bac(0, S0_prime, K, np.sqrt(1 - rho**2)*v0[i], T)

    return np.asarray(Bacs)

def rawMC(S0, K, sigma0, n, m, T, volupdate=None, volupdateparams=None,rho=0):
    brownB=np.random.normal(0,1,(n,m+1))
    brownW=np.random.normal(0,1,(n,m+1))
    dt = np.sqrt(T/m)
    assetpaths = np.zeros((n,m+1))
    assetpaths[:,0] = S0
    sigma1=volupdate(n, m, T, brownB, sigma0, volupdateparams)
    for i in range(m):
        assetpaths[:, i+1] = assetpaths[:,i]+sigma1[:,i]*(rho*brownW[:,i]*dt + np.sqrt(1-(rho**2))*brownB[:,i]*dt)

    return np.asarray(assetpaths[:,m])-K

def simulate_RAW_parallel(id):
    return (id, rawMC(700, 698.5, 0.43, 1500, 2500, 5, volupdate=Heston_Volatility2, volupdateparams={'kappa': 5, 'theta': 0.46, 'vega': 0.6}, rho=0.3))

def simulate_COND_parallel(id):
    return (id, CondMC(700, 698.5, 0.43, 1500, 2500, 5, volupdate=Heston_Volatility2, volupdateparams={'kappa': 5, 'theta': 0.46, 'vega': 0.6}, rho=0.3))

tasks = [i for i in range(12)]

print("starting raw")
rawstart = time.time()
resultsRAW = Parallel(n_jobs=16)(delayed(simulate_RAW_parallel)(i) for (i) in tasks)
payoffraw = []
for id, payraw in resultsRAW:
    payoffraw=np.concatenate((payoffraw, payraw))
rawend = time.time()
rawtime = rawend - rawstart
#----------------------------------------------------------------------
print("starting Cond")
condstart = time.time()
resultsCOND = Parallel(n_jobs=16)(delayed(simulate_COND_parallel)(i) for (i) in tasks)
payoffcond = []
for id, paycond in resultsCOND:
    payoffcond=np.concatenate((payoffcond, paycond))
condend = time.time()
condtime = condend-condstart

pdmethods = {'Method':["Parallel Raw Monte Carlo", "Parallel Conditional Monte Carlo"], 'Average Price':[np.mean(payoffraw), np.mean(payoffcond)], 'Prices Standard Deviation':[np.var(payoffraw), np.var(payoffcond)], 'Exectution Time':[rawtime, condtime]}
pdmethods = pd.DataFrame(pdmethods)
print(payoffraw)
print('--->Raw: ',np.mean(payoffraw), 'std:', np.var(payoffraw))
print('--->Cond:',np.mean(payoffcond), 'std:', np.var(payoffcond))
pdmethods.to_csv('prixSimsResults/parallelmethods.csv')


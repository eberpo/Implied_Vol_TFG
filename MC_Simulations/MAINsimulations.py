import numpy as np
from scipy.stats import norm
import pandas as pd
from joblib import Parallel, delayed

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
    v0 = np.sum(((sigma1[:, :])**2) * dt, axis=1)/ T 

    for j in range(0,m):
        #int_dw=int_dw+sigma1[:, j]*(brownB[:,j+1] - brownB[:, j])*np.sqrt(T/m)
        int_dw=int_dw+sigma1[:, j]*(brownB[:,j+1] - brownB[:, j])*np.sqrt(dt)  

    for i in range(n):
        S0_prime = S0 + rho*(int_dw[i])
        Bacs[i] = Bac(0, S0_prime, K, np.sqrt(1 - rho**2)*v0[i], T)

    return np.mean(Bacs)

incrementsK = 40
incrementsT = 30

Strikes = []
k1 = 695
strinc = 20/incrementsK
for i in range(incrementsK):
    Strikes.append(k1)
    k1 = k1+strinc

Tijms = []
d1 = 1460
tinc = (365*5)/incrementsT
for i in range(incrementsT):
    Tijms.append(d1/365)
    d1 = d1 + tinc    

#parallelization and exection
def simulate_price_parallel(i, j, T, K, F, sigma0, volup, volup_params, rhoin):
    tau = T[i]
    k = K[j]
    price = CondMC(F, k, sigma0, 1500, 1800, tau, volupdate=volup, volupdateparams=volup_params, rho=rhoin)
    return (j, i, price)

simparams=[[{'kappa': 5, 'theta': 0.2, 'vega': 0.3}, 0.3],  [{'kappa': 5, 'theta': 0.2, 'vega': 0.3}, 0.0],  
           [{'kappa': 5, 'theta': 0.8, 'vega': 0.3}, 0.3], [{'kappa': 5, 'theta': 0.8, 'vega': 0.3}, 0.0], 
           [{'kappa': 5, 'theta': 0.46, 'vega': 0.6}, 0.3], [{'kappa': 5, 'theta': 0.46, 'vega': 0.6}, 0.0]]

for simnum, params in enumerate(simparams):
    if simnum > 1 and simnum < 4:
        F = 700
        tasks = [(i, j) for i in range(incrementsT) for j in range(incrementsK)]
        print('--->Starting Parallel Tasks; num of tasks =', incrementsT*incrementsK)
        results = Parallel(n_jobs=16)(delayed(simulate_price_parallel)(i, j, Tijms, Strikes, F, 0.43, Heston_Volatility2, params[0], params[1]) for (i, j) in tasks)
        PricesSIM = np.zeros((incrementsK, incrementsT))
        for j, i, price in results:
            PricesSIM[j][i] = price
        PricesSIM = pd.DataFrame(PricesSIM)
        naam= str('prixSimsResults/CSimSurfaceResults' + str(simnum) + '.csv')
        print(naam)
        PricesSIM.to_csv(naam)


print(CondMC(700, 699.5, 0.43, 1500, 1800, 5, volupdate=Heston_Volatility2, volupdateparams={'kappa': 5, 'theta': 0.46, 'vega': 0.6}, rho=0.3))
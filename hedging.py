import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib import collections as mc
import seaborn as sns

T = 1
steps = 90
sims = 10_000
S0 = 100
mu = 0.1
vol = 0.2
r = 0.02
dband = 0.1
absband = [0.01, 0.99]         
Sfee = 0.005  
Ofee = 0.01
K = 100
putmat = 0.25
callmat = 0.5

def sim(S0 = S0, mu = mu, sig = vol, r = r, T = T, Ndt = steps, sims = sims):
    # simulate sims number of asset prices under S0, mu, and sigma assumptions
    S = np.zeros((sims, Ndt+1))
    S[:,0] = S0
    t = np.linspace(0,T,Ndt + 1)
    dt = T/Ndt
    
    for i in range(sims):
        for j in range(Ndt):
            dW = np.sqrt(dt) * np.random.randn(1)
            S[i,j+1] = S[i,j] * np.exp((mu-0.5*sig**2)*dt + sig*dW )

    return t, S

def call(S = S0, K = K, t = callmat, r = r, sig = vol):
    # call valuation under BS
    dp = (np.log(S/K) + (r+0.5*sig**2)*t) / (sig*np.sqrt(t))
    dm = (np.log(S/K) + (r-0.5*sig**2)*t) / (sig*np.sqrt(t))
    
    return S * norm.cdf(dp) - K*np.exp(-r*t) * norm.cdf(dm)

def Dcall(S = S0, K = K, t = callmat, r = r, sig = vol):
    # delta of call function under BS
    dp = (np.log(S/K) + (r+0.5*sig**2)*t) / (sig*np.sqrt(t))
    
    return norm.cdf(dp)

def put(S = S0, K = K, t = putmat, r = r, sig = vol):
    # put valuation under BS
    dp = (np.log(S/K) + (r+0.5*sig**2)*t) / (sig*np.sqrt(t))
    dm = (np.log(S/K) + (r-0.5*sig**2)*t) / (sig*np.sqrt(t))
    
    return K*np.exp(-r*t) * norm.cdf(-dm) - S * norm.cdf(-dp)

def Dput(S = S0, K = K, t = putmat, r = r, sig = vol):
    # delta of call function under BS
    dp = (np.log(S/K) + (r+0.5*sig**2)*t) / (sig*np.sqrt(t))
    
    return -norm.cdf(-dp)

def gamma(S = S0, K = K, t = steps, r = r, sig = vol):
    # gamma of both call and put functions under BS
    dp = (np.log(S/K) + (r+0.5*sig**2)*t) / (sig*np.sqrt(t))
    
    return norm.pdf(dp) / (S*sig*np.sqrt(t))

def CVaR(sample, quantile = 0.1):
    percent = np.quantile(sample, quantile)
    cvar = sample[sample <= percent].mean()
    return cvar

#%% Discrete time based delta hedging

def dailydeltaput(S0 = S0, K = K, putmat = 0.25, r = r, vol = vol, sims = sims, steps = steps, mu = mu, Sfee = Sfee, Ofee = Ofee):
    # emulates daily discrete time delta hedging
    # Ofee is probably not used i dunno
    
    putprice = put(S0, K, putmat, r, vol)
    
    time, asset = sim(S0 = S0, mu = mu, sig = vol, r = r, T = putmat, Ndt = steps, sims = sims)
    dt = putmat / steps

    bank = np.zeros((sims, steps + 1))
    strat = np.zeros((sims, steps + 1))
    
    strat[:, 0] = Dput(asset[:, 0], K, putmat, r, vol)
    bank[:, 0] = putprice - Ofee*1 - strat[:, 0]*asset[:, 0] - Sfee*np.abs(strat[:, 0])
    
    risk_free = np.exp(r*dt)

    for i in range(1, steps):
        
        strat[:, i] = Dput(asset[:, i], K, putmat - time[i], r, vol)
        diff = strat[:, i] - strat[:, i-1]
        decision = diff * asset[:, i]
        transaction = np.abs(diff) * Sfee
        bank[:, i] = bank[:, i-1] * risk_free - decision - transaction
        
    payoff = K - asset[:, -1]
    payoff[payoff < 0] = 0
    
    # Financial settlement
    bank[:,-1] = bank[:,-2]* risk_free + strat[:,-2] * asset[:,-1] - np.abs(strat[:,-2]) * Sfee
    
    pnl = bank[:, -1] - payoff
    
    
    return time, asset, strat, bank, pnl

#%% discrete test

timeline, price_paths, daily_alpha, bank_history, pnl = dailydeltaput()

plt.plot(timeline[:-1], daily_alpha[1:8,:-1].T)
plt.title(r"$\Delta$ Based Hedging Position at time $t$", fontsize = 16)
plt.xlabel("Years", fontsize = 16)
plt.ylabel(r"$\alpha$ held assets of $S$", fontsize = 16)
plt.show()
#%% Stock Price Path

plt.ylim(75, 135)
plt.xlim(0, 0.25)
plt.plot(timeline[:], price_paths[:2000].T, color = "black", linewidth = 0.1, alpha = 0.3)
plt.plot(timeline[:], price_paths[1:8].T)
plt.title(r"Price Paths at time $t$", fontsize = 16)
plt.xlabel("Years", fontsize = 16)
plt.ylabel(r"Price of Underlying Asset", fontsize = 16)
plt.show()

#%% move based delta hedging

def movedeltaput(S = S0, K = K, putmat = 0.25, sig = vol, sims = sims, steps = steps, mu = mu, Sfee = Sfee, Ofee = Ofee, dband = dband, absband = absband):
    
    putprice = put(S0, K, putmat, r, vol)
    
    time, asset = sim(S0 = S0, mu = mu, sig = vol, r = r, T = putmat, Ndt = steps, sims = sims)
    dt = putmat / steps
    
    bank = np.zeros((sims, steps + 1))
    strat = np.zeros((sims, steps + 1))
    bands = np.zeros((sims*2,steps + 1))
    delta = np.zeros((sims, steps + 1))
    
    strat[:, 0] = Dput(asset[:, 0], K, putmat, r, vol)
    bank[:, 0] = putprice - Ofee*1 - strat[:, 0]*asset[:, 0] - Sfee*np.abs(strat[:, 0])
    
    risk_free = np.exp(r*dt)
    
    sup = absband[1]
    inf = absband[0]
    
    # set up decision bands
    top = strat[:, 0] + dband/2
    bottom = strat[:, 0] - dband/2
    
    bands[::2, 0] = top
    bands[1::2, 0] = bottom
    delta[:, 0] = Dput(asset[:, 0], K, putmat, r, vol)
    
    for i in range(1, steps):
        # calculate current day delta and see if it's moved past the band and still within maximum band
        curr_delta = Dput(asset[:, i], K, putmat - time[i], r, sig)
        delta[:, i] = curr_delta
        
        # check to see if delta has moved past decision band and within absolute boundary
        moved = np.logical_and(np.less(curr_delta, top), np.greater(curr_delta, bottom))
        within = np.logical_and(np.less(strat[:, i], sup), np.greater(strat[:, i], inf))
        moved = np.logical_and(moved, ~within)
        
        # update alpha and bands based on if delta's moved past the band
        strat[:, i] = np.where(~moved, curr_delta, strat[:, i-1])
        top = np.where(moved, top, strat[:, i] + dband/2)
        bottom = np.where(moved, bottom, strat[:, i] - dband/2)
        
        # update bank to reflect alpha change (if any)
        diff = strat[:, i] - strat[:, i-1]
        decision = diff * asset[:, i]
        transaction = np.abs(diff) * Sfee
        risk_free = np.exp(r*dt)
        bank[:, i] = bank[:, i-1] * risk_free - decision - transaction
        
        # update band history
        bands[::2, i] = top
        bands[1::2, i] = bottom
        
    payoff = K - asset[:, -1]
    payoff[payoff < 0] = 0
    
    # Financial settlement
    bank[:,-1] = bank[:,-2]*risk_free + strat[:,-2]*asset[:,-1] - np.abs(strat[:,-2])*Sfee
    
    pnl = bank[:, -1] - payoff
    
    
    return time, asset, strat, bank, pnl, bands, delta

#%% move test

timeline_m, price_paths_m, daily_alpha_m, bank_history_m, pnl_m, bands, delta_hist_m = movedeltaput()

plt.plot(timeline_m[:], daily_alpha_m[1:8,:].T)
plt.title(r"Move Based $\Delta$ Hedging Position at time $t$", fontsize = 16)
plt.xlabel("Years", fontsize = 16)
plt.ylabel(r"$\alpha$ held assets of $S$", fontsize = 16)
plt.show()


bottom = []
pos = bands[0,0]
start = 0
for i in range(bands[0,:].shape[0]):
    if bands[0,:][i] != pos:
        bottom.append([(start, pos), (timeline_m[i], pos)])
        pos = bands[0,i]
        start = timeline_m[i]
       
top = []
pos = bands[1,0]
start = 0
for i in range(bands[1,:].shape[0]):
    if bands[1,:][i] != pos:
        top.append([(start, pos), (timeline_m[i], pos)])
        pos = bands[1,i]
        start = timeline_m[i]

fig, axes = plt.subplots()

axes.set_ylim(-1, 0)

axes.plot(timeline_m[:- 1], daily_alpha_m[0,: -1].T)
axes.plot(timeline_m[: -1], delta_hist_m[0,: -1].T, alpha = 0.5)

bottom = mc.LineCollection(bottom, color = "green", linewidth = 1, alpha = 0.5)
top = mc.LineCollection(top, color = "red", linewidth = 1, alpha = 0.5)
axes.add_collection(bottom)
axes.add_collection(top)

fig.suptitle(r"Move Based $\Delta$ Hedging With $0.1$ Bands", fontsize = 16)
axes.set_xlabel("Years", fontsize = 16)
axes.set_ylabel(r"$\alpha$ held assets of $S$", fontsize = 16)

axes.hlines(-0.99, 0, 0.25, color = "purple", linestyles = "dotted", linewidth = 0.5)
axes.hlines(-0.01, 0, 0.25, color = "purple", linestyles = "dotted", linewidth = 0.5)

axes.legend([ "Held position", r"$\Delta$ History",  "Upper Bound", "Lower Bound", "0.99 and 0.01"])

fig.show()


#%% PNL Function comparison delta only hedging

plt.hist(pnl, density = True, bins = np.linspace(-2,2,30), alpha = 0.3, color = "blue")
plt.hist(pnl_m, density = True, bins = np.linspace(-2,2,30), alpha = 0.3, color = "orange")

dcvar = CVaR(pnl, 0.1)
mcvar = CVaR(pnl_m, 0.1)
plt.axvline(dcvar, color = "blue", alpha = 0.3)
plt.axvline(mcvar, color = "orange", alpha = 0.3)
# sns.kdeplot(pnl_)
plt.title(r"Profits and Losses of $\Delta$ Moved Based V.S. Daily Rebalancing", fontsize = 16)
plt.xlabel("Profit and Loss", fontsize = 16)
plt.ylabel("Frequency", fontsize = 16)
plt.legend([r"Daily $CVaR_{0.1}$= " + str(dcvar)[:5], r"Move $CVaR_{0.1}$= " + str(mcvar)[:5], 'Daily', 'Move-based'], fontsize = 16)
plt.show()

#%% Band Comparison 

colours = ["red", "blue", "green", "orange", "purple"]
bandsizes = [0.2, 0.15, 0.1, 0.05, 0.005]
pnls = []
plt.figure(figsize = (10, 6))
plt.xlim(-2, 2)
for i in bandsizes:
    timeline_m, price_paths_m, daily_alpha_m, bank_history_m, pnl_m, bands, delta_hist_m = movedeltaput(dband = i)
    pnls.append(pnl_m)
    sns.kdeplot(pnl_m, color = colours[bandsizes.index(i)])

sns.kdeplot(pnl, color = "black")

plt.title(r"Comparison of Band Size On Moved Based $\Delta$ Hedging", fontsize = 16)
plt.xlabel("Profits and Losses")
plt.legend(["0.2", "0.15", "0.1", "0.05", "0.005", "Daily"])


    

#%% daily delta gamma hedging

def dailydeltagammaput(S = S0, K = K, callmat = callmat, putmat = putmat, r = r, sig = vol, sims = sims, steps = steps, mu = mu, Ofee = Ofee, Sfee = Sfee):

    put_price = put(S, K, putmat, r, sig)
    dt = putmat/steps
    
    time, asset = sim(S0 = S, mu = mu, sig = sig, r = r, T = putmat, Ndt = steps, sims = sims)
    bank = np.zeros((sims, steps + 1))
    option = np.zeros((sims, steps + 1))
    curr_g = np.zeros((sims, steps))
    curr_d = np.zeros((sims, steps))
    
    ###########################################################################
    # time 0 calculations
    
    option[:,0] = call(S, K, callmat, r, sig)
    curr_g[:,0] = gamma(S, K, putmat, r, sig) / gamma(S, K, callmat, r, sig)
    curr_d[:,0] =  Dput(S, K, putmat, r, sig) - curr_g[:,0]*Dcall(S, K, callmat, r, sig)
    bank[:,0] = put_price - Ofee - curr_d[:,0] * S  - option[:,0]*curr_g[:,0] - np.abs(curr_g[:,0]) * Ofee- np.abs(curr_d[:,0]) * Sfee
    
    risk_free = np.exp(r*dt)

    for i in range(1,steps):
        
        # calculate current option pricing        
        option[:,i] = call(asset[:,i], K, callmat-time[i], r, sig)
        
        # find today's metrics
        curr_g[:,i] = gamma(asset[:,i], K, putmat-time[i], r, sig) / gamma(asset[:,i], K, callmat-time[i], r, sig)
        curr_d[:,i] = Dput(asset[:,i], K, putmat-time[i], r, sig) - curr_g[:,i]*Dcall(asset[:,i], K, callmat-time[i], r, sig)
        
        # update portfolio to calculated metrics
        d_strat = (curr_d[:,i] - curr_d[:,i-1]) * asset[:,i]
        g_strat = (curr_g[:,i] - curr_g[:,i-1]) * option[:,i]
        transaction = np.abs((curr_d[:,i]-curr_d[:,i-1])) * Sfee + np.abs((curr_g[:,i]-curr_g[:,i-1])) * Ofee
        
        # update bank account to portfolio updates
        bank[:,i] = bank[:,i-1]* risk_free - d_strat - g_strat  - transaction

    ##########################################################################
    
    option[:,steps] = call(asset[:,steps], K, callmat-putmat, r, sig)
    
    # liquidate all assets for financial settlement call J.G. Wentworth now
    bank[:,steps] = bank[:,steps-1]*risk_free + curr_d[:,steps-1] * asset[:,steps] + curr_g[:,steps-1] * option[:,steps] - np.abs(curr_g[:,steps-1]) * Ofee - np.abs(curr_d[:,steps-1]) * Sfee
    
    # calculate pnl
    payoff = K - asset[:,-1]
    payoff[(payoff<0)] = 0
    pnl = bank[:,steps] - payoff
    
    return time, asset, option, curr_d, curr_g, bank, pnl


#%% daily delta gamma test
np.random.seed(2022)
time_dg, price_paths_dg, option_path_dg, daily_alpha_dg, daily_gamma_dg, bank_history_dg, pnl_dg = dailydeltagammaput()

 
plt.plot(time_dg[:-1], daily_alpha_dg[1:8,:].T)
plt.title(r"Unit of Asset in Daily $\Delta ,\Gamma$ Hedge", fontsize = 16)
plt.xlabel("Time in years", fontsize = 16)
plt.ylabel("Unit of Stock", fontsize = 16)
plt.show()
    

plt.plot(time_dg[:-1], daily_gamma_dg[1:8,:].T)
plt.title(r"Units of Call Options in Daily $\Delta, \Gamma$ Hedge", fontsize = 16)
plt.xlabel("Time in years", fontsize = 16)
plt.ylabel("Unit of Call Option", fontsize = 16)
plt.show()


plt.hist(pnl_dg, bins = np.linspace(-0.7,0,20), alpha = 0.3, density = True)
plt.xlim(-0.7, 0.3)
plt.title(r"Profits and Losses of Daily Rebalancing, $\Delta, \Gamma$ Strategy", fontsize = 16)
plt.xlabel("Profit and Loss", fontsize = 16)
plt.ylabel("Frequency", fontsize = 16)
plt.legend(['daily', 'move-based'], fontsize = 16)
plt.show()

#%% move based delta gamma hedging

def movedeltagammaput(S = S0, K = K, callmat = 0.5, putmat = 0.25, r = r, sig = vol, sims = sims, steps = steps, mu = mu, Sfee = Sfee, Ofee = Ofee, dband = dband, absband = absband):
    
    put_price = put(S, K, putmat, r, sig)
    dt = putmat/steps
    
    time, asset = sim(S0 = S, mu = mu, sig = sig, r = r, T = putmat, Ndt = steps, sims = sims)
    bank = np.zeros((sims, steps + 1))
    option = np.zeros((sims, steps + 1))
    curr_g = np.zeros((sims, steps))
    curr_d = np.zeros((sims, steps))
    
    moved_g = np.zeros((sims, steps + 1))
    moved_d = np.zeros((sims, steps + 1))
    

    bands = np.zeros((sims*2,steps + 1))
    ###########################################################################
    # time zero calculations
    
    option[:,0] = call(S, K, callmat, r, sig)
    curr_g[:,0] = gamma(S, K, putmat, r, sig) / gamma(S, K, callmat, r, sig)
    curr_d[:,0] = Dput(S, K, putmat, r, sig) - curr_g[:,0]*Dcall(S, K, callmat, r, sig)
    bank[:,0] = put_price - Ofee - curr_d[:,0] * S  - option[:,0]*curr_g[:,0] - np.abs(curr_g[:,0]) * Ofee- np.abs(curr_d[:,0]) * Sfee
    
    
    moved_g[:, 0] = np.copy(curr_g[:,0])
    moved_d[:, 0] = np.copy(curr_d[:,0])
    
    
    risk_free = np.exp(r*dt)
    
    top = curr_d[:, 0] + dband/2
    bottom = curr_d[:, 0] - dband/2
    
    bands[::2, 0] = top
    bands[1::2, 0] = bottom
    
    ###########################################################################
    # time 1 to 89 calculations
    
    for i in range(1, steps):
        
        # calculate current option pricing        
        option[:,i] = call(asset[:,i], K, callmat-time[i], r, sig)
        
        # find today's metrics
        curr_g[:,i] = gamma(asset[:,i], K, putmat-time[i], r, sig) / gamma(asset[:,i], K, callmat-time[i], r, sig)
        curr_d[:,i] = Dput(asset[:,i], K, putmat-time[i], r, sig) - curr_g[:,i]*Dcall(asset[:,i], K, callmat-time[i], r, sig)
        
        # check if metrics have changed
        moved = np.logical_and(np.less(curr_d[:, i], top), np.greater(curr_d[:, i], bottom))
        
        moved_d[:, i] = np.where(~moved, curr_d[:, i], moved_d[:, i-1])
        moved_g[:, i] = np.where(~moved, curr_g[:, i], moved_g[:, i-1])
        
        # update portfolio to calculated metrics
        d_strat = (moved_d[:,i] - moved_d[:,i-1]) * asset[:,i]
        g_strat = (moved_g[:,i] - moved_g[:,i-1]) * option[:,i]
        transaction = np.abs((moved_d[:,i]-moved_d[:,i-1])) * Sfee + np.abs((moved_g[:,i]-moved_g[:,i-1])) * Ofee
        
        # update bank account to portfolio updates
        bank[:,i] = bank[:,i-1]* risk_free - d_strat - g_strat  - transaction
        
        # update bands
        top = np.where(moved, top, moved_d[:, i] + dband/2)
        bottom = np.where(moved, bottom, moved_d[:, i] - dband/2)
        
        bands[::2, i] = top
        bands[1::2, i] = bottom
        
    ##########################################################################

    option[:,steps] = call(asset[:,steps], K, callmat-putmat, r, sig)
    
    # liquidate all assets for financial settlement call J.G. Wentworth now
    bank[:,steps] = bank[:,steps-1]*risk_free + moved_d[:,steps-1] * asset[:,steps] + moved_g[:,steps-1] * option[:,steps] - np.abs(moved_g[:,steps-1]) * Ofee - np.abs(moved_d[:,steps-1]) * Sfee
    
    # calculate pnl
    payoff = K - asset[:,steps]
    payoff[(payoff<0)] = 0
    pnl = bank[:,steps] - payoff
        
    return time, asset, option, moved_d, moved_g, curr_d, curr_g, bank, bands, pnl


#%%
        

time_dgm, price_paths_dgm, option_path_dgm, daily_alpha_dgm, daily_gamma_dgm, true_d, true_g, bank_history_dgm, dg_bands, pnl_dgm = movedeltagammaput()

 
plt.plot(time_dgm[:-1], daily_alpha_dgm[1:10,:-1].T)
plt.title(r"Unit of Asset in Move Based $\Delta ,\Gamma$ Hedge", fontsize = 16)
plt.xlabel("Time in years", fontsize = 16)
plt.ylabel("Unit of Stock", fontsize = 16)
plt.show()
    

plt.plot(time_dgm[:-1], daily_gamma_dgm[1:10,:-1].T)
plt.title(r"Units of Call Options in Move Based $\Delta, \Gamma$ Hedge", fontsize = 16)
plt.xlabel("Time in years", fontsize = 16)
plt.ylabel("Unit of Call Option", fontsize = 16)
plt.show()
            
        
        
        
bottom = []
pos = dg_bands[0,0]
start = 0
for i in range(dg_bands[0,:].shape[0]):
    if dg_bands[0,:][i] != pos:
        bottom.append([(start, pos), (time_dgm[i], pos)])
        pos = dg_bands[0,i]
        start = time_dgm[i]
       
top = []
pos = dg_bands[1,0]
start = 0
for i in range(dg_bands[1,:].shape[0]):
    if dg_bands[1,:][i] != pos:
        top.append([(start, pos), (time_dgm[i], pos)])
        pos = dg_bands[1,i]
        start = time_dgm[i]

fig, axes = plt.subplots()

axes.plot(time_dgm[:- 1], daily_alpha_dgm[0,: -1].T)
axes.plot(time_dgm[: -1], true_d[0,:].T, alpha = 0.5)

bottom = mc.LineCollection(bottom, color = "green", linewidth = 1, alpha = 0.5)
top = mc.LineCollection(top, color = "red", linewidth = 1, alpha = 0.5)
axes.add_collection(bottom)
axes.add_collection(top)

fig.suptitle(r"Move Based $\Delta, \Gamma$ Hedging With $0.1$ Bands", fontsize = 16, y = 0.95)
axes.set_xlabel("Years", fontsize = 16)
axes.set_ylabel(r"$\alpha$ held assets of $S$", fontsize = 16)
axes.legend([ "Held position", r"$\Delta$ History",  "Upper Bound", "Lower Bound"])
fig.show()

#%% PnL Comparions delta gamma daily vs move


dcvar = CVaR(pnl_dg, 0.1)
mcvar = CVaR(pnl_dgm, 0.1)
plt.axvline(dcvar, color = "blue", alpha = 0.3)
plt.axvline(mcvar, color = "orange", alpha = 0.3)
    
plt.hist(pnl_dg, density = True, bins = np.linspace(-0.7,0.5,30), alpha = 0.3, color = "blue")
plt.hist(pnl_dgm, density = True, bins = np.linspace(-0.7,0.5,30), alpha = 0.3, color = "orange")
plt.title(r"Profits and Losses of Moved Based V.S. Daily Rebalancing, $\Delta, \Gamma$ Model", fontsize = 16)
plt.xlabel("Profit and Loss", fontsize = 16)
plt.ylabel("Frequency", fontsize = 16)
plt.legend([r"Daily $CVaR_{0.1}$= " + str(dcvar)[:5], r"Move $CVaR_{0.1}$= " + str(mcvar)[:5], 'Daily', 'Move-based'], fontsize = 16, loc = 1)
plt.show()

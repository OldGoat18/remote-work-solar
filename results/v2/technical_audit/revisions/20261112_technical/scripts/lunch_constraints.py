"""Registered lunch scenarios: interval energy, strict [11:00,14:00).

Slots 22..27 are indices 5..10 of the 20-slot activity arrays. Half of each
source interval may move; an out-of-window destination retains that energy
at its source. The original lunch peak is an explicit engineering cap.
"""
import numpy as np
from numba import njit, prange

VERSION = 'strict_lunch_window_peak_20261114'
SHIFTS = np.arange(-2, 3)
ORDER = (2, 1, 3, 0, 4)  # ties keep original, then shorter advance/delay

@njit
def candidate_lunch(C):
    n=C.shape[0]
    profiles=np.zeros((n,5,6));valid=np.ones((n,5),np.bool_)
    moved=np.zeros((n,5));retained=np.zeros((n,5))
    for i in range(n):
        peak=0.
        for t in range(6):peak=max(peak,C[i,t+5])
        for k in range(5):
            shift=k-2
            for t in range(6):profiles[i,k,t]=.5*C[i,t+5]
            for t in range(6):
                amount=.5*C[i,t+5];dest=t+shift
                if dest<0 or dest>=6:
                    dest=t;retained[i,k]+=amount
                elif shift!=0:moved[i,k]+=amount
                profiles[i,k,dest]+=amount
            for t in range(6):
                if profiles[i,k,t]>peak+max(1e-14,1e-12*peak):valid[i,k]=False
        # Identity rule is stored exactly, avoiding rounding-only differences.
        for t in range(6):profiles[i,2,t]=C[i,t+5]
        valid[i,2]=True
    return profiles,valid,moved,retained

def prepare(B,C):
    profiles,valid,moved,retained=candidate_lunch(np.ascontiguousarray(C))
    demand=B[:,None,5:11]-C[:,None,5:11]+profiles
    demand[:,2,:]=B[:,5:11]
    fixed=np.where(valid[:,3],3,2).astype(np.int64)
    return np.ascontiguousarray(demand),valid,moved,retained,fixed

@njit
def one_gain(supply,candidates,valid,fixed):
    best=0.;choice=2;fixed_gain=0.
    for k in ORDER:
        if k==2 or not valid[k]:continue
        gain=0.
        for t in range(6):
            gain+=min(supply[t],candidates[k,t])-min(supply[t],candidates[2,t])
        if k==fixed:fixed_gain=gain
        if gain>best+1e-12:best=gain;choice=k
    return best,fixed_gain,choice

@njit
def evaluate_surplus(D,candidates,valid,moved,retained,fixed):
    """D is mesh x six lunch slots x eligible dates; returns paired gains."""
    nd=D.shape[2];gains=np.zeros((nd,2));counts=np.zeros((nd,5),np.int64)
    movement=np.zeros((nd,4))  # upper moved/retained; fixed moved/retained
    for d in range(nd):
        for i in range(D.shape[0]):
            upper,rule,k=one_gain(D[i,:,d],candidates[i],valid[i],fixed[i])
            gains[d,0]+=upper;gains[d,1]+=rule;counts[d,k]+=1
            movement[d,0]+=moved[i,k];movement[d,1]+=retained[i,k]
            movement[d,2]+=moved[i,fixed[i]];movement[d,3]+=retained[i,fixed[i]]
    return gains,counts,movement

@njit(parallel=True)
def evaluate_pv(G,L,GI,pi,fac,candidates,valid,moved,retained,fixed):
    """All inputs restricted to six lunch slots and eligible dates.

    fac is path x date x prefecture x lunch-slot. Only existing GI is perturbed.
    """
    totals=np.zeros((len(fac),2));counts=np.zeros((len(fac),5),np.int64)
    movement=np.zeros((len(fac),4))
    for path in prange(len(fac)):
        for d in range(G.shape[2]):
            for i in range(G.shape[0]):
                supply=np.empty(6)
                for t in range(6):
                    generation=max(G[i,t,d]+GI[i,t,d]*fac[path,d,pi[i],t],0.)
                    supply[t]=max(generation-L[i,t,d],0.)
                upper,rule,k=one_gain(supply,candidates[i],valid[i],fixed[i])
                totals[path,0]+=upper;totals[path,1]+=rule;counts[path,k]+=1
                movement[path,0]+=moved[i,k];movement[path,1]+=retained[i,k]
                movement[path,2]+=moved[i,fixed[i]];movement[path,3]+=retained[i,fixed[i]]
    return totals,counts,movement

def lunch_schedule(D,B,C):
    candidates,valid,moved,retained,fixed=prepare(B,C)
    gains,counts,_=evaluate_surplus(np.ascontiguousarray(D[:,5:11,:]),candidates,valid,moved,retained,fixed)
    return np.minimum(D,B[:,:,None]).sum(axis=(0,1))+gains[:,0],counts.sum(axis=0)

def fixed_profile(B,C):
    candidates,valid,moved,retained,fixed=prepare(B,C)
    output=B.copy();output[:,5:11]=candidates[np.arange(len(B)),fixed]
    return output

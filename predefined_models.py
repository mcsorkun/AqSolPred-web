#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 14:15:21 2019

@author: Murat Cihan Sorkun
"""


from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
import numpy as np 
import mordred
from mordred import Calculator, descriptors
from mordred import SLogP,Chi,ABCIndex,BondCount,Polarizability,RingCount,EState,RotatableBond,CarbonTypes,Aromatic,AtomCount,VdwVolumeABC,McGowanVolume,HydrogenBond
from mordred import BertzCT, BalabanJ,EccentricConnectivityIndex



#mlp with 1 test set
def get_errors(y_true,y_pred):   

    err_mae=mae(y_true,y_pred)
    err_rmse=np.sqrt(mse(y_true,y_pred))
    err_r2=r2(y_true,y_pred)
        
    print("Ensemble MAE:"+str(err_mae)+" RMSE:"+str(err_rmse)+" R2:"+str(err_r2))
  
    return err_mae,err_rmse,err_r2
   


#returns mordred descriptor vector
def predefined_mordred(mol, desc_type="best", desc_names=False):
    
    calc1 = mordred.Calculator()    

    if(desc_type in ["best"]):
        calc1.register(mordred.SLogP)
        calc1.register(mordred.HydrogenBond.HBondAcceptor)
        calc1.register(mordred.HydrogenBond.HBondDonor)
        calc1.register(mordred.AtomCount.AtomCount("HeavyAtom"))
        calc1.register(mordred.TopoPSA.TopoPSA(True))
        calc1.register(mordred.RingCount.RingCount(None, False, False, None, None))
        calc1.register(mordred.BondCount.BondCount("any", False))
        
    
    if(desc_type in ["all","atom"]): 
        calc1.register(mordred.AtomCount.AtomCount("X"))
        calc1.register(mordred.AtomCount.AtomCount("HeavyAtom"))
        calc1.register(mordred.Aromatic.AromaticAtomsCount)
        

    if(desc_type in ["all","bond"]):  
        calc1.register(mordred.HydrogenBond.HBondAcceptor)
        calc1.register(mordred.HydrogenBond.HBondDonor)
        calc1.register(mordred.RotatableBond.RotatableBondsCount)  
        calc1.register(mordred.BondCount.BondCount("any", False))
        calc1.register(mordred.Aromatic.AromaticBondsCount)  
       	calc1.register(mordred.BondCount.BondCount("heavy", False))
       	calc1.register(mordred.BondCount.BondCount("single", False))
       	calc1.register(mordred.BondCount.BondCount("double", False))
        calc1.register(mordred.BondCount.BondCount("triple", False))

    if(desc_type in ["all","topological"]):      
        calc1.register(mordred.McGowanVolume.McGowanVolume)
        calc1.register(mordred.TopoPSA.TopoPSA(True))
        calc1.register(mordred.TopoPSA.TopoPSA(False))
        calc1.register(mordred.MoeType.LabuteASA)
        calc1.register(mordred.Polarizability.APol)
        calc1.register(mordred.Polarizability.BPol)
        calc1.register(mordred.AcidBase.AcidicGroupCount)
        calc1.register(mordred.AcidBase.BasicGroupCount)
        calc1.register(mordred.EccentricConnectivityIndex.EccentricConnectivityIndex)        
        calc1.register(mordred.TopologicalCharge.TopologicalCharge("raw",1))
        calc1.register(mordred.TopologicalCharge.TopologicalCharge("mean",1))
        
        
    if(desc_type in ["all","index"]): 
        calc1.register(mordred.SLogP)
        calc1.register(mordred.BertzCT.BertzCT)
        calc1.register(mordred.BalabanJ.BalabanJ)
        calc1.register(mordred.WienerIndex.WienerIndex(True))
        calc1.register(mordred.ZagrebIndex.ZagrebIndex(1,1))
        calc1.register(mordred.ABCIndex)
        
    if(desc_type in ["all","ring"]):     
        calc1.register(mordred.RingCount.RingCount(None, False, False, None, None))
        calc1.register(mordred.RingCount.RingCount(None, False, False, None, True))
        calc1.register(mordred.RingCount.RingCount(None, False, False, True, None))
        calc1.register(mordred.RingCount.RingCount(None, False, False, True, True))
        calc1.register(mordred.RingCount.RingCount(None, False, False, False, None))
        calc1.register(mordred.RingCount.RingCount(None, False, True, None, None))
       

    if(desc_type in ["all","estate"]):   
        calc1.register(mordred.EState)
        
# if desc_names is "True" returns only name list
    if(desc_names):
        name_list=[]
        for desc in calc1.descriptors:
            name_list.append(str(desc))
        return name_list
#        return list(calc1._name_dict.keys())
    else: 
        result = calc1(mol)
        return result._values
   


#!/bin/bash

from radonpy.core import utils, poly
from radonpy.ff.gaff2_mod import GAFF2_mod
from radonpy.sim import qm
from radonpy.sim.preset import eq,tc

smiles = '*C(C*)c1ccccc1'
ter_smiles='*C'
temp=300
press = 1.0
omp_psi4=1
mpi=1
omp=1
gpu=0
mem=1000
work_dir ='./'
ff = GAFF2_mod()

if __name__=='__main__':
    #Conformation search
    mol = utils.mol_from_smiles(smiles)
    mol, energy = qm.conformation_search(mol,ff=ff,work_dir=work_dir,psi4_omp=omp_psi4,mpi=mpi,omp=omp,memory=mem,log_name='monomer1')
    #Electronic property calculation
    qm.assign_charges(mol,charge='RESP',opt=False,work_dir=work_dir, omp=omp_psi4,memory=mem, log_name='monomer1')
    qm_data=qm.sp_prop(mol, opt=False,work_dir=work_dir, omp=omp_psi4,memory=mem, log_name='monomer1')
    polar_data=qm.polarizability(mol, opt=False,work_dir=work_dir, omp=omp_psi4,memory=mem, log_name='monomer1')
    ter=utils.mol_from_smiles(iter_smiles)
    qm.assign_charges(ter,charge='RESP',opt=True,work_dir=work_dir, omp=omp_psi4,memory=mem, log_name='ter1')
    dp = poly.calc_n_from_num_atoms(mol,1000,terminal1=ter)
    homopoly=poly.polymerize_rw(mol,dp,tacticity='atactic')
    homopoly=poly.terminate_rw(homopoly,ter)
    result = ff.ff_assign(homopoly)

    if not result:
        print('[ERROR: Cannot assign force field parameters.]')
    ac = poly.amorphous_cell(homopoly,10,density=0.05)
    eqmd = eq.EQ21step(ac,work_dir=work_dir)
    ac=eqmd.exec(temp=temp,press=1.0,mpi=mpi,omp=omp,gpu=gpu)
    analy = eqmd.analyze()
    prop_data = analy.get_all_prop(temp=temp,press=1.0, save=True)
    result = analy.check_eq()
    
    for i in range(4):
        if result:break
        eqmd = eq.Additional(ac,work_dir=work_dir)
        ac=eqmd.exec(temp=temp,press=press,mpi=mpi,omp=omp,gpu=gpu)
        analy=eqmd.analyze()
        prop_data=analy.get_all_prop(temp=temp,press=press,save=True)
        result=analy.check_eq()
    
    if not result:
        print('[ERROR:Did not reach an equilibrium state.]')
    else:
        nemd=tc.NEMD_MP(ac,work_dir=work_dir)
        ac = nemd.exec(decomp=True, temp=temp, mpi=mpi,omp=omp,gpu=gpu)
        nemd_analy = nemd.analyze()
        TC = nemd_analy.calc.tc(decomp=True,save=True)
        if not nemd_analy.Tgrad_data['Tgrad_check']:
            print('[ERROR: low linearity of temperature gradient.]')
        print('Thermal conductivity:%f' %TC)

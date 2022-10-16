import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from time import time
import GraphManager as gm
import igraph as ig
from networkx import Graph as nxGraph

class YSNetModel:
    def __init__(self,G,f):

        if type(G)==list:
            Na,Nnet,L1,L2=gm.getBigGraph(G)

        elif type(G)==ig.Graph:
            L1,L2=gm.getLL(G)
            Nnet=L2.size-1
            Na=1

        #if its networkx graph convert to igraph
        elif type(G)==nxGraph:
            G=ig.Graph.from_networkx(G)
            L1,L2=gm.getLL(G)
            Nnet=L2.size-1
            Na=1

        else:
            raise Exception('Unknown graph type. Use igraph or networkx graph.')
            
        self.Nnet=Nnet
        self.Na=Na
        self.tL1=L1
        self.tL2=L2
        self.N=L2.size-1
        self.f=f
        self.threadsperblock=1024
        self.blockspergrid=self.Na

        #Alocamos memoria en la GPU para riquezas, riesgos, listas, y semaforos i y j
        self.d_Nwealths=cuda.device_array(self.N,dtype=np.float32)
        self.d_Nwi=cuda.device_array(self.N,dtype=np.float32)
        self.d_Nrisks=cuda.device_array(self.N,dtype=np.float32)
        self.d_L1=cuda.device_array(self.tL1.size,dtype=np.int32)
        self.d_L2=cuda.device_array(self.tL2.size,dtype=np.int32)
        self.d_SI=cuda.device_array(self.N,dtype=np.int32)
        self.d_SJ=cuda.device_array(self.N,dtype=np.int32)

        #semaforos en estado inicial
        SI=np.ones(self.N,dtype=np.int32)
        cuda.to_device(SI,to=self.d_SI)
        SJ=np.ones(self.N,dtype=np.int32)
        cuda.to_device(SJ,to=self.d_SJ)

        #Alocamos estado para números aleatoprios
        self.rng_states = create_xoroshiro128p_states(self.threadsperblock*self.blockspergrid, seed=time())
        self.reset()


    def reset(self,type='uniform',r=0.1):
        if type=='uniform':
            Nwealths=np.random.uniform(0,1,self.N)
            Nrisks=np.random.uniform(0,1,self.N)

        elif type=='constant':
            Nwealths=np.ones(self.N)
            Nrisks=np.ones(self.N)*r

        Nwealths=Nwealths/np.sum(Nwealths)
        for l in range(self.Na):
            Nwealths[l*self.Nnet:(l+1)*self.Nnet]=Nwealths[l*self.Nnet:(l+1)*self.Nnet]/np.sum(Nwealths[l*self.Nnet:(l+1)*self.Nnet])
        Nwealths=Nwealths.astype(np.float32)
        cuda.to_device(Nwealths,to=self.d_Nwealths)
        Nrisks=Nrisks.astype(np.float32)    
        cuda.to_device(Nrisks,to=self.d_Nrisks)

        #Disponemos los vecinos en la GPU (son fijas para todos los f)
        cuda.to_device(self.tL1.astype(np.int32),to=self.d_L1)
        cuda.to_device(self.tL2.astype(np.int32),to=self.d_L2)

    def getWealths(self):
        return self.d_Nwealths.copy_to_host()

    def getRisks(self):
        return self.d_Nrisks.copy_to_host()

    def setRisks(self,R):
        cuda.to_device(R.astype(np.float32),to=self.d_Nrisks)

    def setWealths(self,W):
        cuda.to_device(W.astype(np.float32),to=self.d_Nwealths)

    def setRisk(self,A,r):
        Nrisks=self.d_Nrisks.copy_to_host()
        Nrisks[A]=r
        Nrisks=Nrisks.astype(np.float32)
        cuda.to_device(Nrisks,to=self.d_Nrisks)

    def setWealth(self,A,w):
        Nwealths=self.d_Nwealths.copy_to_host()
        Nwealths[A]=w
        Nwealths=Nwealths.astype(np.float32)
        cuda.to_device(Nwealths,to=self.d_Nwealths)

    def loadNeigh(self,L1,L2):
        self.tL1=L1
        self.tL2=L2
        self.d_L1=cuda.device_array(self.tL1.size,dtype=np.int32)
        self.d_L2=cuda.device_array(self.tL2.size,dtype=np.int32)
        cuda.to_device(self.tL1.astype(np.int32),to=self.d_L1)
        cuda.to_device(self.tL2.astype(np.int32),to=self.d_L2)

    def termalize(self,M):
        gpu_MCS[self.blockspergrid,self.threadsperblock](self.d_Nwealths,self.d_Nrisks,self.d_SI,self.d_SJ,self.f,self.d_L1,self.d_L2,self.rng_states,M,self.Nnet,self.Na)
        cuda.synchronize()

    def epoch(self,M):
        Nwi=np.zeros(self.N)
        Nwi=Nwi.astype(np.float32)
        cuda.to_device(Nwi,to=self.d_Nwi)

        gpu_MCSplus[self.blockspergrid,self.threadsperblock](self.d_Nwealths,self.d_Nrisks,self.d_SI,self.d_SJ,self.f,self.d_L1,self.d_L2,self.rng_states,M,self.Nnet,self.Na,self.d_Nwi)

        return self.d_Nwi.copy_to_host()/M
    
    def follow(self,M,agent):

        Wi=np.zeros(M)
        Wi=Wi.astype(np.float32)
        d_Wi=cuda.device_array(M,dtype=np.float32)
        cuda.to_device(Wi,to=d_Wi)
        gpu_MCSfollow[self.blockspergrid,self.threadsperblock](self.d_Nwealths,self.d_Nrisks,self.d_SI,self.d_SJ,self.f,self.d_L1,self.d_L2,self.rng_states,M,self.Nnet,self.Na,d_Wi,agent)
        d_Wi.copy_to_host(Wi)
        del d_Wi

        return Wi

    def getGini():
        #calculate gini coefficient
        pass
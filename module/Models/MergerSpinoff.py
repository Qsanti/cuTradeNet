import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states
from .Kernels.kMS import gpu_MCS,gpu_MCSfollow,gpu_MCSplus
from time import time
from .Utils import GraphManager as gm
import igraph as ig
from networkx import Graph as nxGraph

class MSNetModel:
    def __init__(self,G):
        '''Create a new YS model with the given graph or list of graphs and f'''

        if type(G)==list:
            if type(G[0])==nxGraph:
                G=[ig.Graph.from_networkx(g) for g in G]
            Na,Nnet,L1,L2=gm.getBigGraph(G)


        elif type(G)==ig.Graph:
            L1,L2=gm.toLL(G)
            Nnet=L2.size-1
            Na=1

        #if its networkx graph convert to igraph
        elif type(G)==nxGraph:
            G=ig.Graph.from_networkx(G)
            L1,L2=gm.toLL(G)
            Nnet=L2.size-1
            Na=1

        else:
            raise Exception('Unknown graph type. Use igraph or networkx graph.')
            
        self.Nnet=Nnet
        self.Na=Na
        self.tL1=L1
        self.tL2=L2
        self.N=L2.size-1
        self.f=0
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


    def reset(self,wealth_type='uniform',risk_type='hetereogeneus',r=0.1):
        '''
        Reset the model to random state in risks and wealths. 
        wealth_type: 'uniform' or 'equal'
        risk_type:  'hetereogeneus' or 'homogeneus'
        r: if risk_type is 'homogeneus' this is the risk for all agents
        '''

        if risk_type=='hetereogeneus':
            Nrisks=np.random.uniform(0,1,self.N)
        elif risk_type=='homogeneus':
            Nwealths=np.ones(self.N)
            Nrisks=np.ones(self.N)*r
        else:
            raise Exception('''Unsupported risk type. Use 'hetereogeneus' or 'homogeneus'.''')

        if wealth_type=='uniform':
            Nwealths=np.random.uniform(0,1,self.N)
        elif wealth_type=='equal':
            Nwealths=np.ones(self.N)
        else:
            raise Exception('''Unsupported wealth type. Use 'uniform' or 'equal'.''')

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
        '''Return the wealths of the agents'''
        return self.d_Nwealths.copy_to_host()

    def getRisks(self):
        '''Return the risks of the agents'''
        return self.d_Nrisks.copy_to_host()

    def setRisks(self,R):
        '''Set the risks of the agents
        R: array of risks in the same order as the agents'''
        cuda.to_device(R.astype(np.float32),to=self.d_Nrisks)

    def setWealths(self,W):
        '''Set the wealths of the agents
        W: array of wealths in the same order as the agents'''
        cuda.to_device(W.astype(np.float32),to=self.d_Nwealths)

    def setRisk(self,A,r):
        '''Set the risk of the agents indexed by A to r
        A: indexes of the agents Ex: [1,2,3]
        r: risk to set or array of risks Ex: 0.1 or [0.1,0.2,0.3]''' 
        R=self.getRisks()
        R[A]=r
        self.setRisks(R)
        Nrisks=self.d_Nrisks.copy_to_host()
        Nrisks[A]=r
        Nrisks=Nrisks.astype(np.float32)
        cuda.to_device(Nrisks,to=self.d_Nrisks)

    def setWealth(self,A,w):
        '''Set the wealth of the agents indexed by A to w
        A: indexes of the agents Ex: [1,2,3]
        w: wealth to set or array of wealths Ex: 0.1 or [0.1,0.2,0.3]''' 
        Nwealths=self.d_Nwealths.copy_to_host()
        Nwealths[A]=w
        Nwealths=Nwealths.astype(np.float32)
        cuda.to_device(Nwealths,to=self.d_Nwealths)

    def modifyGraph(self,G):
        '''Modify the graph of the model'''
        pass
      
    def termalize(self,M):
        '''Termalize the model for M montecarlo steps
        M: number of montecarlo steps'''
        gpu_MCS[self.blockspergrid,self.threadsperblock](self.d_Nwealths,self.d_Nrisks,self.d_SI,self.d_SJ,self.f,self.d_L1,self.d_L2,self.rng_states,M,self.Nnet,self.Na)
        cuda.synchronize()

    def epoch(self,M):
        '''Make an epoch of M montecarlo steps returning the mean temporal wealths in each agent
        M: number of montecarlo steps'''
        Nwi=np.zeros(self.N)
        Nwi=Nwi.astype(np.float32)
        cuda.to_device(Nwi,to=self.d_Nwi)

        gpu_MCSplus[self.blockspergrid,self.threadsperblock](self.d_Nwealths,self.d_Nrisks,self.d_SI,self.d_SJ,self.f,self.d_L1,self.d_L2,self.rng_states,M,self.Nnet,self.Na,self.d_Nwi)

        return self.d_Nwi.copy_to_host()/M
    
    def follow(self,M,agent):
        '''Make an epoch of M montecarlo steps returning the wealths of the agent in each step
        M: number of montecarlo steps
        agent: index of the agent'''
        Wi=np.zeros(M)
        Wi=Wi.astype(np.float32)
        d_Wi=cuda.device_array(M,dtype=np.float32)
        cuda.to_device(Wi,to=d_Wi)
        gpu_MCSfollow[self.blockspergrid,self.threadsperblock](self.d_Nwealths,self.d_Nrisks,self.d_SI,self.d_SJ,self.f,self.d_L1,self.d_L2,self.rng_states,M,self.Nnet,self.Na,d_Wi,agent)
        d_Wi.copy_to_host(Wi)
        del d_Wi

        return Wi

    def getGini(self):
        '''Return the Gini coefficient of the actual wealth distribution'''
        x=self.d_Nwealths.copy_to_host()
        #divide x in self.Na parts
        x=x.reshape(self.Na,self.Nnet)

        #sort each part
        sorted_x = np.sort(x,axis=1)
        #calculate the cumulative sum of each part
        cumx = np.cumsum(sorted_x, dtype=np.float32, axis=1)


        return (self.Nnet + 1 - 2 * np.sum(cumx,axis=1) / cumx[:,-1]) / self.Nnet

        
        pass
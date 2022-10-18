import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states
from . Kernels.kYS import gpu_MCS,gpu_MCSfollow,gpu_MCSplus
from time import time
from . Utils import GraphManager as gm
import igraph as ig
from networkx import Graph as nxGraph
import warnings


class YSNetModel:
    '''Class for a Yard Sale model on complex networks.
    Runs transactions in GPU using numba, multiple graphs can be used running in parallel.
    '''
    
    def __init__(self,G,f: float,wmin=1e-17):
        '''Create a new YS model with the given graph or list of graphs and f value
        G: igraph/ntworkx graph or list of igraph/networkx graphs
        f: social protection factor
        wmin: minimum wealth an agent has to have to be able to transact'''

        if f>0.5 or f<0:
            raise Exception('social protection factor f must be between 0 and 0.5')

        if wmin<0:
            raise Exception('Minimum wealth to transact must be positive')

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

        if Nnet>1024:
            raise Exception('Graphs cannot be bigger than 1024 nodes for GPU compatibility')
            
        self.__Nnet=Nnet
        self.__Na=Na
        self.__tL1=L1
        self.__tL2=L2
        self.__N=L2.size-1
        self.__f=f
        self.__wmin=wmin
        self.__threadsperblock=1024
        self.__blockspergrid=self.__Na

        #Alocamos memoria en la GPU para riquezas, riesgos, listas, y semaforos i y j
        self.__d_Nwealths=cuda.device_array(self.__N,dtype=np.float32)
        self.__d_Nwi=cuda.device_array(self.__N,dtype=np.float32)
        self.__d_Nrisks=cuda.device_array(self.__N,dtype=np.float32)
        self.__d_L1=cuda.device_array(self.__tL1.size,dtype=np.int32)
        self.__d_L2=cuda.device_array(self.__tL2.size,dtype=np.int32)
        self.__d_SI=cuda.device_array(self.__N,dtype=np.int32)
        self.__d_SJ=cuda.device_array(self.__N,dtype=np.int32)

        #semaforos en estado inicial
        SI=np.ones(self.__N,dtype=np.int32)
        cuda.to_device(SI,to=self.__d_SI)
        SJ=np.ones(self.__N,dtype=np.int32)
        cuda.to_device(SJ,to=self.__d_SJ)

        #Alocamos estado para números aleatoprios
        self.__rng_states = create_xoroshiro128p_states(self.__threadsperblock*self.__blockspergrid, seed=time())
        self.reset()

    def __str__(self) -> str:
        return f'Yard Sale model: \nGraph: {self.__Na} graphs of {self.__Nnet} agents \nSocial protection factor f: {self.__f}\nMinimum wealth to transact: {self.__wmin}'
        

    def reset(self,wealth_type='uniform',risk_type='hetereogeneous',r=0.1):
        '''
        Reset the model to random state in risks and wealths. 
        wealth_type: 'uniform' or 'equal'
        risk_type:  'hetereogeneous' or 'homogeneous'
        r: if risk_type is 'homogeneous' this is the risk for all agents
        '''

        if risk_type=='hetereogeneous':
            Nrisks=np.random.uniform(0,1,self.__N)
        elif risk_type=='homogeneous':
            Nwealths=np.ones(self.__N)
            Nrisks=np.ones(self.__N)*r
        else:
            raise Exception('''Unsupported risk type. Use 'hetereogeneous' or 'homogeneous'.''')

        if wealth_type=='uniform':
            Nwealths=np.random.uniform(0,1,self.__N)
        elif wealth_type=='equal':
            Nwealths=np.ones(self.__N)
        else:
            raise Exception('''Unsupported wealth type. Use 'uniform' or 'equal'.''')

        Nwealths=Nwealths/np.sum(Nwealths)

        for l in range(self.__Na):
            Nwealths[l*self.__Nnet:(l+1)*self.__Nnet]=Nwealths[l*self.__Nnet:(l+1)*self.__Nnet]/np.sum(Nwealths[l*self.__Nnet:(l+1)*self.__Nnet])
        Nwealths=Nwealths.astype(np.float32)
        cuda.to_device(Nwealths,to=self.__d_Nwealths)
        Nrisks=Nrisks.astype(np.float32)    
        cuda.to_device(Nrisks,to=self.__d_Nrisks)

        #Disponemos los vecinos en la GPU (son fijas para todos los f)
        cuda.to_device(self.__tL1.astype(np.int32),to=self.__d_L1)
        cuda.to_device(self.__tL2.astype(np.int32),to=self.__d_L2)


    @property
    def f(self):
        '''Return the social protection factor f'''
        return self.__f

    @f.setter
    def f(self,f):
        '''Set the social protection factor f'''
        if f>0.5 or f<0:
            raise Exception('social protection factor f must be between 0 and 0.5')
        self.__f=f

    @property
    def wmin(self):
        '''Return the minimum wealth to transact'''
        return self.__wmin

    @wmin.setter
    def wmin(self,wmin):
        '''Set the minimum wealth to transact'''
        if wmin<0:
            raise Exception('Minimum wealth to transact must be positive')
        self.__wmin=wmin

    def get_wealths(self):
        '''Return the wealths of the agents'''
        return self.__d_Nwealths.copy_to_host()

    def get_risks(self):
        '''Return the risks of the agents'''
        return self.__d_Nrisks.copy_to_host()

    def get_graphSize(self) -> tuple:
        '''Return a tuple with the number of graphs and the number of agents per graph'''
        return (self.__Nnet,self.__Na)


    def get_graph(self):
        '''Return the graph of the model as igraph graph'''
        return gm.toGraph(self.__tL1,self.__tL2)

    def get_nxGraph(self):
        '''Return the graph of the model as networkx graph'''
        return gm.toGraph(self.__tL1,self.__tL2).to_networkx()

    def __getGraphs(self): #todo
        '''Return the graphs of the model as list of igraph graphs'''
        #return gm.getGraphs(self.__tL1,self.__tL2)
        pass

    def set_risks(self,R):
        '''Set the risks of the agents
        R: array of risks in the same order as the agents'''
        cuda.to_device(R.astype(np.float32),to=self.__d_Nrisks)

    def set_wealths(self,W):
        '''Set the wealths of the agents
        W: array of wealths in the same order as the agents'''
        cuda.to_device(W.astype(np.float32),to=self.__d_Nwealths)

    def set_risk(self,A,r):
        '''Set the risk of the agents indexed by A to r
        A: indexes of the agents Ex: [1,2,3]
        r: risk to set or array of risks Ex: 0.1 or [0.1,0.2,0.3]''' 
        R=self.get_risks()
        R[A]=r
        self.setRisks(R)
        Nrisks=self.__d_Nrisks.copy_to_host()
        Nrisks[A]=r
        Nrisks=Nrisks.astype(np.float32)
        cuda.to_device(Nrisks,to=self.__d_Nrisks)

    def set_wealth(self,A,w):
        '''Set the wealth of the agents indexed by A to w
        A: indexes of the agents Ex: [1,2,3]
        w: wealth to set or array of wealths Ex: 0.1 or [0.1,0.2,0.3]''' 
        Nwealths=self.__d_Nwealths.copy_to_host()
        Nwealths[A]=w
        Nwealths=Nwealths.astype(np.float32)
        cuda.to_device(Nwealths,to=self.__d_Nwealths)

    def __modifyGraph(self,G):
        '''Modify the graph of the model'''
        pass
      
    def termalize(self,M : int):
        '''Termalize the model for M montecarlo steps
        M: number of montecarlo steps'''
        warnings.simplefilter('ignore')
        gpu_MCS[self.__blockspergrid,self.__threadsperblock](self.__d_Nwealths,self.__d_Nrisks,self.__d_SI,self.__d_SJ,self.__f,self.__wmin,self.__d_L1,self.__d_L2,self.__rng_states,M,self.__Nnet,self.__Na)
        cuda.synchronize()
        warnings.simplefilter('default')

    def epoch(self,M : int):
        '''Make an epoch of M montecarlo steps returning the mean temporal wealths in each agent
        M: number of montecarlo steps'''
        Nwi=np.zeros(self.__N)
        Nwi=Nwi.astype(np.float32)
        cuda.to_device(Nwi,to=self.__d_Nwi)
        warnings.simplefilter('ignore')
        gpu_MCSplus[self.__blockspergrid,self.__threadsperblock](self.__d_Nwealths,self.__d_Nrisks,self.__d_SI,self.__d_SJ,self.__f,self.__wmin,self.__d_L1,self.__d_L2,self.__rng_states,M,self.__Nnet,self.__Na,self.__d_Nwi)
        cuda.synchronize()
        warnings.simplefilter('default')
        return self.__d_Nwi.copy_to_host()/M
    
    def follow(self,M : int ,agent : int):
        '''Make an epoch of M montecarlo steps returning the wealths of the agent in each step
        M: number of montecarlo steps
        agent: index of the agent'''
        Wi=np.zeros(M)
        Wi=Wi.astype(np.float32)
        d_Wi=cuda.device_array(M,dtype=np.float32)
        cuda.to_device(Wi,to=d_Wi)
        warnings.simplefilter('ignore')
        gpu_MCSfollow[self.__blockspergrid,self.__threadsperblock](self.__d_Nwealths,self.__d_Nrisks,self.__d_SI,self.__d_SJ,self.__f,self.__wmin,self.__d_L1,self.__d_L2,self.__rng_states,M,self.__Nnet,self.__Na,d_Wi,agent)
        cuda.synchronize()
        warnings.simplefilter('default')
        d_Wi.copy_to_host(Wi)
        del d_Wi

        return Wi

    def Gini(self):
        '''Return the Gini coefficient of the actual wealth distribution'''
        x=self.__d_Nwealths.copy_to_host()
        #divide x in self.__Na parts
        x=x.reshape(self.__Na,self.__Nnet)

        #sort each part
        sorted_x = np.sort(x,axis=1)
        #calculate the cumulative sum of each part
        cumx = np.cumsum(sorted_x, dtype=np.float32, axis=1)


        return (self.__Nnet + 1 - 2 * np.sum(cumx,axis=1) / cumx[:,-1]) / self.__Nnet

        

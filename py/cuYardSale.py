import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from time import time


class YSNetModel:
    def __init__(self,Nnet,L1,L2,f):
        self.Nnet=Nnet
        self.N=L2.size-1
        self.Na=self.N//Nnet
        self.tL1=L1
        self.tL2=L2
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


    



@cuda.jit
def gpu_MCS(Nw,Nr,SI,SJ,f,L1,L2,rng_states,M,N,Na):
    
    idx=cuda.threadIdx.x 
    bidx=cuda.blockIdx.x

    
    if idx<N and bidx<Na:
        i=bidx*N+idx
    
        if L2[i+1]-L2[i]==0: #si no tiene vecinos no hay intercambios
            SI[i]=0 

        else:
            for m in range(M):
                
                j=L1[L2[i]:L2[i+1]][int(xoroshiro128p_uniform_float32(rng_states, i)*(L2[i+1]-L2[i]))]

                SI[i]=1 #hilo activa
                SJ[i]=j #indica el j con el que cambia
                cuda.syncthreads() #esperamos que todos los hilos lo carguen

            
            
                
                for i2 in range(bidx*N,i): #le prestamos atención a hilos menores
                    #chequeamos si esta activo
                    while(cuda.atomic.compare_and_swap(SI[i2:i2+1],-1,10)): #cambio absurdo para leer el valor atomicamente

                        if(i2==j or SJ[i2]==i or SJ[i2]==j): #si se superpone j con i, i con j o j con j
                            continue #esperalo 
                            
                        break #pasa al siguiente
                            
                #obtenemos las riquezas de los agentes
                wi=Nw[i] 
                wj=Nw[j]


                if wi>3e-14 and wj>3e-14:

                    if wi*Nr[i]<wj*Nr[j]:
                        dw=wi*Nr[i]
                    else:
                        dw=wj*Nr[j]

                    
                    pn=0.5+f*(wj-wi)/(wi+wj)
                    x=pn/(1-pn)
                    n=int(bool(int(xoroshiro128p_uniform_float32(rng_states, i)*(1+x))))

                    
                    if n:
                        Nw[i]=Nw[i]+dw
                        Nw[j]=Nw[j]-dw
                    else:
                        Nw[i]=Nw[i]-dw
                        Nw[j]=Nw[j]+dw

                    
                        
                
                SI[i]=0 #liberamos
                cuda.syncthreads() #esperamos a todos los hilos del bloque


        
@cuda.jit
def gpu_MCSplus(Nw,Nr,SI,SJ,f,L1,L2,rng_states,M,N,Na,Wis):
    
    idx=cuda.threadIdx.x 
    bidx=cuda.blockIdx.x

    
    if idx<N and bidx<Na:
        i=bidx*N+idx
    
        if L2[i+1]-L2[i]==0: #si no tiene vecinos no hay intercambios
            SI[i]=0 
            cuda.atomic.add(Wis[i:i+1],0,Nw[i]*M)



        else:
            for m in range(M):
                
                j=L1[L2[i]:L2[i+1]][int(xoroshiro128p_uniform_float32(rng_states, i)*(L2[i+1]-L2[i]))]

                SI[i]=1 #hilo activa
                SJ[i]=j #indica el j con el que cambia
                cuda.syncthreads() #esperamos que todos los hilos lo carguen

            
            
                
                for i2 in range(bidx*N,i): #le prestamos atención a hilos menores
                    #chequeamos si esta activo
                    while(cuda.atomic.compare_and_swap(SI[i2:i2+1],-1,10)): #cambio absurdo para leer el valor atomicamente

                        if(i2==j or SJ[i2]==i or SJ[i2]==j): #si se superpone j con i, i con j o j con j
                            continue #esperalo 
                            
                        break #pasa al siguiente
                            
                #obtenemos las riquezas de los agentes
                wi=Nw[i] 
                wj=Nw[j]


                if wi>3e-14 and wj>3e-14:

                    if wi*Nr[i]<wj*Nr[j]:
                        dw=wi*Nr[i]
                    else:
                        dw=wj*Nr[j]

                    
                    pn=0.5+f*(wj-wi)/(wi+wj)
                    x=pn/(1-pn)
                    n=int(bool(int(xoroshiro128p_uniform_float32(rng_states, i)*(1+x))))

                    
                    if n:
                        Nw[i]=Nw[i]+dw
                        Nw[j]=Nw[j]-dw
                    else:
                        Nw[i]=Nw[i]-dw
                        Nw[j]=Nw[j]+dw

                    
                        
                
                SI[i]=0 #liberamos
                cuda.syncthreads() #esperamos a todos los hilos del bloque

                #actualizamos suma de la riqueza de los agentes
                cuda.atomic.add(Wis[i:i+1],0,Nw[i])



@cuda.jit
def gpu_MCSfollow(Nw,Nr,SI,SJ,f,L1,L2,rng_states,M,N,Na,Wis,agent):
    
    idx=cuda.threadIdx.x 
    bidx=cuda.blockIdx.x

    
    if idx<N and bidx<Na:
        i=bidx*N+idx
    
        if L2[i+1]-L2[i]==0: #si no tiene vecinos no hay intercambios
            SI[i]=0 

        else:
            for m in range(M):
                
                j=L1[L2[i]:L2[i+1]][int(xoroshiro128p_uniform_float32(rng_states, i)*(L2[i+1]-L2[i]))]

                SI[i]=1 #hilo activa
                SJ[i]=j #indica el j con el que cambia
                cuda.syncthreads() #esperamos que todos los hilos lo carguen

            
            
                
                for i2 in range(bidx*N,i): #le prestamos atención a hilos menores
                    #chequeamos si esta activo
                    while(cuda.atomic.compare_and_swap(SI[i2:i2+1],-1,10)): #cambio absurdo para leer el valor atomicamente

                        if(i2==j or SJ[i2]==i or SJ[i2]==j): #si se superpone j con i, i con j o j con j
                            continue #esperalo 
                            
                        break #pasa al siguiente
                            
                #obtenemos las riquezas de los agentes
                wi=Nw[i] 
                wj=Nw[j]


                if wi>3e-14 and wj>3e-14:

                    if wi*Nr[i]<wj*Nr[j]:
                        dw=wi*Nr[i]
                    else:
                        dw=wj*Nr[j]

                    
                    pn=0.5+f*(wj-wi)/(wi+wj)
                    x=pn/(1-pn)
                    n=int(bool(int(xoroshiro128p_uniform_float32(rng_states, i)*(1+x))))

                    
                    if n:
                        Nw[i]=Nw[i]+dw
                        Nw[j]=Nw[j]-dw
                    else:
                        Nw[i]=Nw[i]-dw
                        Nw[j]=Nw[j]+dw

                    
                        
                
                SI[i]=0 #liberamos
                cuda.syncthreads() #esperamos a todos los hilos del bloque

                
                if i==agent:
                    Wis[m]=Nw[i]

            
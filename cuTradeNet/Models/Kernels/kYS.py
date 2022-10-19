from numba import cuda
from numba.cuda.random import xoroshiro128p_uniform_float32



@cuda.jit
def gpu_MCS(Nw,Nr,SI,SJ,f,wmin,L1,L2,rng_states,M,N,Na):
    
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


                if wi>wmin and wj>wmin:

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
def gpu_MCSepoch(Nw,Nr,SI,SJ,f,wmin,L1,L2,rng_states,M,N,Na,Wis):
    
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


                if wi>wmin and wj>wmin:

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
def gpu_MCSfollow(Nw,Nr,SI,SJ,f,wmin,L1,L2,rng_states,M,N,Na,Wis,agent):
    
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


                if wi>wmin and wj>wmin:

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

            
# cuTradeNet

**cuTradeNet** is a library that provides classes to easily create & run *kinetic wealth exchange models* on *complex networks*. 

Leads the user to set one (or ensemble) of *complex networks* as a contact structure agents use to trade about. Using the following wealth exchange models:
* [Yard-sale model](https://www.sciencedirect.com/science/article/pii/S0378437120309237 "model details here")
* [Merger-Spinoff model](https://www.sciencedirect.com/science/article/pii/S0378437120309237 "model details here")

It is written in Python and uses Cuda module form the Numba package to accelerate the simulation runnin in the GPU, *paralelizing some transaccions* in the same graph and *paralelizing runs* in multiple graphs leading faster outcome estimations.
It's completely abstracted from the CUDA knowledge for the user, so you can use it as a regular Python library.


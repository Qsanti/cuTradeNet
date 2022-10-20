# cuTradeNet

**cuTradeNet** is a library that provides classes to easily create & run [*kinetic wealth exchange models*](https://rf.mokslasplius.lt/elementary-kinetic-exchange-models/ "online mini simulations") on [*complex networks*](https://en.wikipedia.org/wiki/Complex_network "complex networks wiki"). 

Leads the user to set one (or ensemble) of *complex networks* as a contact structure agents use to trade about. The following wealth exchange models were implemented:
* [Yard-sale model](https://www.sciencedirect.com/science/article/pii/S0378437120309237 "model details here")
* [Merger-Spinoff model](https://www.sciencedirect.com/science/article/pii/S0378437120309237 "model details here")

It is written in Python and uses Cuda module from [Numba](https://numba.pydata.org/ "Numba page") package to accelerate the simulation runnin in GPU, *paralelizing some transaccions* in the same graph and *paralelizing runs* in multiple graphs, leading to  easier & faster averaging of system properties.
It's completely abstracted from the CUDA knowledge for the user, so you can use it as a regular Python library.

### How to use
There is a [Demo notebook](https://github.com/Qsanti/cuTradeNet/Models&Demo) in the repository that can be tryed in it's [Google Colab]() version too.
There is also a Gereneral explanation of Kinetic Wealth Exchange Models in [this notebook](https://github.com/Qsanti/cuTradeNet/Models&Demo/ModelsList).
'''
For creating a new model use:
> Models.ModelName(*args,**kwargs)
where ModelName is the name of the model and *args and **kwargs are the arguments of the model.

Or import the model and use:
> from cuTradeNet.Models import ModelName
> ModelName(*args,**kwargs)

Example:
> from cuTradeNet import Models
> S=Models.YardSale(G=g,f=0.3)

List of avaliable models by name:
* YardSale - Yard Sale model
* MergerSpinoff - Merger Spinoff model
* Constant - Constant exchange model
* DragulescuYakovenko - Drăgulescu and Yakovenko model
* ChakrabortiChakrabarti -  Chakraborti and Chakrabarti model
* Allin - "All in" trade model

Especifications of the models at:
https://github.com/Qsanti/cuTradeNet/blob/master/Models%26Demo/ModelsList.ipynb

'''


from .YardSale import YSNetModel as YardSale
from .MergerSpinoff import MSNetModel as MergerSpinoff
from .Constant import CNetModel as Constant
from .DragulescuYakovenko import DYNetModel as DragulescuYakovenko
from .ChakrabortiChakrabarti import CCNetModel as ChakrabortiChakrabarti 
from .Allin import AiNetModel as Allin



import numpy as np

class Qtable:
    "Crea Q table segun entradas y salidas pedidas discoun y learning rate"
    def __init__(self, n_states, n_actions, low_reward=-2,disc=0.95,learn=0.1):
        self.table = np.random.uniform(low=low_reward, high=0, size=(n_states + [n_actions]))
        self.discount = disc
        self.learning_rate = learn

    def action(self, state):
        "Return accion that maximize Q value of table"
        return np.argmax(self.table[state])

    def update(self, state, action, reward, new_state):
        current_q = self.table[state][action]
        max_future_q = np.max(self.table[new_state])
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount * max_future_q)
        self.table[state][action] = new_q

    def get(self):
        return self.table

    def save(self, file_name):
        np.save(file_name, self.table)

    def color_table(self):
        return np.argmax(self.table, axis=2)

    def scale_table(self):
        #difference btwn values in axis 2
        return self.table[:,:,1]-self.table[:,:,0]

    





class Epsilon_gen:
    def __init__(self,episodes,initial=1,start_epsilon_dec=1,episodes_dec=0.5):

        self.epsilon=initial
        self.start_epsilon_dec=start_epsilon_dec
        self.stop_epsilon_dec=np.floor(episodes*episodes_dec)
        self.dec=initial/( self.stop_epsilon_dec - start_epsilon_dec)
        self.episode=0

    def __call__(self):

        if self.episode>=self.start_epsilon_dec and self.episode<self.stop_epsilon_dec-1:
            self.epsilon-=self.dec
        if self.episode==self.stop_epsilon_dec-1:
            self.epsilon=0
        self.episode+=1
        return self.epsilon


        

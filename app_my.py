import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import streamlit as st
plt.rcParams['figure.dpi'] = 300

class Schelling_model:
    def __init__(self, size, empty_percent, similarity_threshold):
        self.size = size
        self.empty_percent = empty_percent
        self.similarity_threshold = similarity_threshold
        self.n_iterations = n_iterations

        self.side = int(np.sqrt(size))
        self.size = self.side ** 2
        p = [(1-empty_percent)/2, (1-empty_percent)/2, empty_percent]
        self.city = np.random.choice([-1, 1, 0], size=self.size, p=p)
        self.city = np.reshape(self.city, (self.side, self.side))

    def get_bounds(self, x):
        a, b = x, x + 1
        if x != 0:
            a -= 1
        if x != self.side:
            b += 1
        return a, b
        
    def iter(self):
        empty_houses = list(zip(np.where(self.city == 0)[0], np.where(self.city == 0)[1]))
        old_houses = []
        old_agents = []
        for (row, col), value in np.ndenumerate(self.city):
            agent = self.city[row, col]
            if agent != 0:
                row_bounds = self.get_bounds(row)
                col_bounds = self.get_bounds(col)
                neighbours = self.city[row_bounds[0]:row_bounds[1], col_bounds[0]:col_bounds[1]]
                agent_like_neighbours= len(np.where(neighbours == agent)[0]) - 1
                similarity = agent_like_neighbours / (np.size(neighbours) - 1)
                if (similarity < self.similarity_threshold):
                    old_houses.append((row, col))
                    old_agents.append(agent)
        numb_want_to_move = len(old_houses)
        houses_to_move = empty_houses + old_houses
        new_houses = random.sample(houses_to_move, numb_want_to_move)

        for i in range(numb_want_to_move):
            self.city[old_houses[i]] = 0
        for i in range(numb_want_to_move):
            self.city[new_houses[i]] = old_agents[i]
            
        return numb_want_to_move

    def get_numb_want_to_move(self):
        numb_want_to_move = 0
        for (row, col), value in np.ndenumerate(self.city):
            agent = self.city[row, col]
            if agent != 0:
                row_bounds = self.get_bounds(row)
                col_bounds = self.get_bounds(col)
                neighbours = self.city[row_bounds[0]:row_bounds[1], col_bounds[0]:col_bounds[1]]
                agent_like_neighbours= len(np.where(neighbours == agent)[0]) - 1
                similarity = agent_like_neighbours / (np.size(neighbours) - 1)
                if (similarity < self.similarity_threshold):
                    numb_want_to_move += 1
        return numb_want_to_move
        


#Streamlit App

st.title("Schelling's Model of Segregation")

population_size = st.sidebar.slider("Population Size", 500, 10000, 2500)
empty_precent = st.sidebar.slider("Percent of empty houses", 0., 1., 0.2)
similarity_threshold = st.sidebar.slider("Similarity Threshold", 0., 1., .4)
n_iterations = st.sidebar.number_input("Number of Iterations", 10)

schelling = Schelling_model(population_size, empty_precent, similarity_threshold)
numb_over_time = []
numb_over_time.append(schelling.get_numb_want_to_move())

#Plot the graphs at initial stage
plt.style.use("seaborn-v0_8")
plt.figure(figsize=(8, 4))

# Left hand side graph with Schelling simulation plot
cmap = ListedColormap(['red', 'white', 'dodgerblue'])
plt.subplot(121)
plt.axis('off')
plt.pcolor(schelling.city, cmap=cmap, edgecolors='w', linewidths=1)

# Right hand side graph with Number over time graph
plt.subplot(122)
plt.xlabel("Iterations")
plt.xlim([0, n_iterations])
plt.title("Number of houses want to move", fontsize=15)

city_plot = st.pyplot(plt)

progress_bar = st.progress(0)

if st.sidebar.button('Run Simulation'):

    for i in range(n_iterations):
        numb_want_to_move = schelling.iter()
        numb_over_time.append(numb_want_to_move)
        plt.figure(figsize=(8, 4))
    
        plt.subplot(121)
        plt.axis('off')
        plt.pcolor(schelling.city, cmap=cmap, edgecolors='w', linewidths=1)

        plt.subplot(122)
        plt.xlabel("Iterations")
        plt.xlim([0, n_iterations])
        plt.title("Number of houses want to move", fontsize=15)
        plt.plot(range(1, len(numb_over_time)+1), numb_over_time)

        city_plot.pyplot(plt)
        plt.close("all")
        progress_bar.progress((i+1.)/n_iterations)
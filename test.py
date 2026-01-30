from application import Application
from agent_ql import AgentQL
from environment import Environment

myEnv = Environment(10, 10)
myAgent = AgentQL(
    myEnv, 15000, 0.005, 0.05, 1.0, 1.0/15000, 100
)
myApp = Application(myEnv, myAgent)


myApp.launch()

# width = 5

# pos = width**2 // 2

# move = [-width, width, -1, 1]

# state =[\
#     "a0","a1","a2","a3","a4",\
#     "b0","b1","b2","b3","b4",\
#     "c0","c1","c2","c3","c4",\
#     "d0","d1","d2","d3","d4",\
#     "e0","e1","e2","e3","e4"]

# idx = 2
# print(state[int(pos + idx * move[0])])
# print(state[int(pos + idx * move[1])])
# print(state[int(pos + idx * move[2])])
# print(state[int(pos + idx * move[3])])




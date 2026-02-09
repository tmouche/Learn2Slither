from application import Application
from agent_ql import AgentQL
from environment import Environment

myEnv = Environment(10, 10)
myAgent = AgentQL(
    myEnv, 1000000, 0.005, 0.05, 1.0, 1.0/15000, 10000
)
myApp = Application(myEnv, myAgent)


myApp.launch()






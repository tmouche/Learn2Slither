from application import Application
from agent_ql import AgentQL
from environment import Environment

myEnv = Environment(10, 10)
myAgent = AgentQL(
    env=myEnv,
    epoch=15000,
    l_rate=0.05,
    discount=0.5,
    e_rate=1.0,
    e_decay=1.0/15000, 
    max_action=10000,
    path_to_save="__q_matrix.json"
)
# myAgent = AgentQL(
#     env=myEnv, q_matrix_path="__q_matrix.json"
# )
myApp = Application(myEnv, myAgent)



myApp.launch()






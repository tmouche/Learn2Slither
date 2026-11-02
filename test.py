from app.application import Application
from core.agent_ql import AgentQL
from core.environment import Environment
from utils.dataclass import EnvironmentSettings, AgentSettings, HyperParameters, RewardSettings

sett = EnvironmentSettings()
sett.height = 10
sett.width = 10

hp = HyperParameters()
hp.discount = 0.95
hp.exploration_rate = 1.0
hp.learning_rate = 0.25
hp.max_action = 10000
hp.epoch = 5000
hp.exploration_decay = 1.0/hp.epoch

agg = AgentSettings()
agg.agent_name = 'test'
agg.hyper_params = hp
agg.rewards = RewardSettings()
agg.rewards.green_apple = 10.0
agg.rewards.loose = -100.0
agg.rewards.noting = -1
agg.rewards.red_apple = -15
agg.rewards.win = 100.0


myEnv = Environment(sett)
myAgent = AgentQL(
    env=myEnv,
    agent_settings=agg,
)
# myAgent = AgentQL(
#     env=myEnv, q_matrix_path="__q_matrix.json"
# )
# myApp = Application(myEnv, myAgent)


x = myAgent.train_sequentially()
# myApp.launch()

# ATTENTION SI HEIGHT ET WIDTH IMPAIRE ON EST OUT OF RANGE
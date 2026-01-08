from application import Application
from environment import Environment

myEnv = Environment(15, 25)
myApp = Application(myEnv)

myApp.launch()

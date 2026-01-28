from application import Application
from environment import Environment

myEnv = Environment(10, 10)
myApp = Application(myEnv)

myApp.launch()

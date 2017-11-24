import simulator
import matplotlib.pyplot as plt

environment= simulator.Simulator()

for i in range(100):
    environment.step(environment.uniform_sample())
    print(i)
result=environment.gethistory()
plt.plot(result)
plt.show()



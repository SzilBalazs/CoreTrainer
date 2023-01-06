import matplotlib.pyplot as plt

f = open("loss.txt")
x_axis = []
y_axis = []
for line in f.readlines():
    data = line.split(" ")
    x_axis.append(float(data[0]))
    y_axis.append(1-float(data[1]))

fig, ax = plt.subplots()
ax.plot(x_axis, y_axis)
ax.set(xlabel="iteration", ylabel="Loss", title="Training loss")
ax.grid()

fig.savefig("loss.png")

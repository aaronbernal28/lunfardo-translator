import matplotlib.pyplot as plt
import numpy as np
import time

plt.ion()  # Turn on interactive mode

x = np.linspace(0, 2 * np.pi, 10)
y = np.sin(x)

fig, ax = plt.subplots()
line, = ax.plot(x, y, "o-")

for i in range(50):
    new_y = np.sin(x + i * 0.1)
    line.set_ydata(new_y)
    fig.canvas.draw()  # Redraw the canvas
    fig.canvas.flush_events() # Process GUI events
    time.sleep(0.1)

box_text = f'Perplexity final: {33}\nExecution time: {0.1}s'
ax.text(0.7, 0.9, box_text, transform=ax.transAxes)

plt.ioff() # Turn off interactive mode (optional)
plt.savefig('images/example.png')
plt.show() # If you want to keep the final figure open after the loop
from bokeh.plotting import figure, show

x = [1, 2, 3, 4, 5]
y = [6, 7, 2, 4, 5]

# Create a figure for plotting with the figure class constructor
p = figure(title="Simple bokeh example", x_axis_label='x label', y_axis_label='y lbl')

# add a line graph to the figure
p.line(x, y, legend_label="Temp", line_width=1.5)

# Generate the graph and open a web browser to show the figure
show(p)
import ternary
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def plot_ternary(data_points, cover_vector, title, ax):
    scale = 1  # Simplex sum should be 1
    tax = ternary.TernaryAxesSubplot(ax=ax, scale=scale)

    for i, (point, cover) in enumerate(zip(data_points, cover_vector)):
        color = 'red' if cover else 'blue'  # True = Red, False = Blue
        tax.scatter([point], marker='o', color=color, s=50)

    # Configure ternary plot
    tax.boundary(linewidth=1.5)  # Draw the simplex boundary
    tax.gridlines(multiple=0.2, color="gray", linestyle="dotted")  # Grid
    tax.left_axis_label("Component 1", fontsize=12)
    tax.right_axis_label("Component 2", fontsize=12)
    tax.bottom_axis_label("Component 3", fontsize=12)
    tax.ticks(axis='lbr', multiple=0.2, linewidth=1, offset=0.02)
    tax.clear_matplotlib_ticks()  # Remove extra ticks
    ax.set_title(title, fontsize=14)

    

def plot_ternary_size(data_points, cover_vector, title, ax,
                      vmin, vmax,
                      cmap = cm.plasma):
    scale = 1
    tax = ternary.TernaryAxesSubplot(ax=ax, scale=scale)
    scatter = tax.scatter(data_points, marker='o', c=cover_vector, 
                          cmap=cmap, s=50, vmin=vmin, vmax=vmax)

    tax.boundary(linewidth=1.5) 
    tax.gridlines(multiple=0.2, color="gray", linestyle="dotted")
    tax.left_axis_label("Component 1", fontsize=12)
    tax.right_axis_label("Component 2", fontsize=12)
    tax.bottom_axis_label("Component 3", fontsize=12)
    tax.ticks(axis='lbr', multiple=0.2, linewidth=1, offset=0.02)
    tax.clear_matplotlib_ticks()
    ax.set_title(title, fontsize=14)

    return scatter

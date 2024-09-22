# OLG plot
using Plots

# working directory is the current file
cd(dirname(@__FILE__))

# Parameters
J = 10  # Number of periods an agent lives
generations = J  # Number of generations
overlap_time = 0  # Time period where all generations overlap

# Initialize arrays for the scatter plot
x = []  # time t
y = []  # generation (now reflecting birth times)
colors = []  # Color for each square

# Fill the arrays with the coordinates for each generation
for g in 1:generations
    birth_time = -J + g  # The top generation is born at t = -J + 1
    for j in 1:J
        current_time = birth_time + j - 1
        push!(x, current_time)  # Add time period
        push!(y, birth_time)  # Y-axis will now reflect the birth time directly
        
        # Check if the current time is where all generations overlap
        if current_time == overlap_time
            push!(colors, :red)  # Color it red if it's the overlapping column
        else
            push!(colors, :blue)  # Otherwise, color it blue
        end
    end
end

# Create the plot, with y-axis flipped and y values reflecting birth times
gr()
scatter(x, y, marker=:rect, ms=10, xlabel="Time t", ylabel="Birth time of OLG", legend=false, aspect_ratio=:equal, yflip=true, color=colors, xlim=(-J, J), xticks=-J:1:J, yticks=-J+1:0)
savefig("OLGplot")


using Plots

# working directory is the current file
cd(dirname(@__FILE__))

# Parameters
J = 10  # Number of periods an agent lives
generations = J  # Number of generations

# Initialize arrays for the scatter plot
x = []  # x-axis (life periods from 1 to J)
y = []  # generation (now reflecting birth times)
colors = []  # Color for each square

# Fill the arrays with the coordinates for each generation
for g in 1:generations
    birth_time = -J + g  # The top generation is born at t = -J + 1
    for j in 1:J
        push!(x, j)  # Life periods start from 1 instead of 0
        push!(y, birth_time)  # Y-axis will now reflect the birth time directly
        
        # Color the reverse diagonal red (when j matches the generation index g)
        if j == generations - g + 1
            push!(colors, :red)  # Color the reverse diagonal red
        else
            push!(colors, :blue)  # Otherwise, color it blue
        end
    end
end

# Adjust the plot size and marker size to leave white space between the squares
gr()
scatter(x, y, marker=:rect, ms=12,  # Decrease marker size for more white space
       xlabel="Life Period", ylabel="Birth Time of OLG", 
       legend=false, aspect_ratio=:equal, yflip=true, color=colors,
       xlim=(0.5, J + 0.5),  # Adjust the x-axis limits to fit life periods from 1 to J
       ylim=(-J - 0.5, 0.5),  # Keep the slightly expanded y-axis limits
       xticks=1:1:J, yticks=-J+1:0)

# Save the plot
savefig("OLG_life_periods_1_to_J_with_space.png")

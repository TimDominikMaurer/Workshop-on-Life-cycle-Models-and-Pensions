# OLG plot
using Plots

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
scatter(x, y, marker=:rect, ms=10, xlabel="Time t", ylabel="Birth Time", legend=false, aspect_ratio=:equal, yflip=true, color=colors, xlim=(-J, J), xticks=-J:1:J, yticks=-J+1:0)

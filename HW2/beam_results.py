import matplotlib.pyplot as plt

beam_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9]

f1 = [
    76.67,
    84.12,
    86.52,
    87.69,
    88.05,
    88.05,
    88.05,
    88.05,
    88.05]

t = [10.212757,
     13.532603,
     16.880476,
     20.560129,
     23.357027,
     27.026674,
     29.289614,
     32.379262,
     35.864751]

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(beam_sizes, f1)
ax1.set_title("Accuracy (F1)")
ax1.set_xlabel("Beam Size")
ax1.set_ylabel("F1 Score")

ax2.plot(beam_sizes, t)
ax2.set_title("Runtime")
ax2.set_xlabel("Beam Size")
ax2.set_ylabel("Runtime (seconds)")

plt.show()
# plt.savefig(f"./sse-plots/momentum-{lr}.png")

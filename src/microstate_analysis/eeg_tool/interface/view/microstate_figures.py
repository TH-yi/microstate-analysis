import matplotlib.pyplot as plt

fig = plt.figure(dpi=80)
ax = fig.add_subplot(1,1,1)
table_data=[
    ["matched gt", 10],
    ["unmatched gt", 20],
    ["total gt", 30],
    ["mean_precision", 0.6],
    ["mean_recall", 0.4]
]
table = ax.table(cellText=table_data, loc='center')
table.set_fontsize(14)
table.scale(1,4)
ax.axis('off')
plt.show()
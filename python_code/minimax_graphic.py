import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()

# Add nodes with labels
G.add_node('a', label='1')
G.add_node('b', label='1')
G.add_node('c', label='≤ -4') 
G.add_node('d', label='1')
G.add_node('e', label='2')
G.add_node('f', label='-4')
G.add_node('g', label='?')

# Add edges
G.add_edge('a', 'b')
G.add_edge('a', 'c')
G.add_edge('b', 'd')
G.add_edge('b', 'e')
G.add_edge('c', 'f')
G.add_edge('c', 'g')

white_nodes = ['a', 'd', 'e', 'f', 'g']
black_nodes = ['b', 'c']

pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(10, 6))

# Manually specify positions to match text layout
pos = {
    'a': (0, 2),
    'b': (-1, 1),
    'c': (1, 1),
    'd': (-1.5, 0),
    'e': (-0.5, 0),
    'f': (0.5, 0),
    'g': (1.5, 0)
}

nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='->', arrowsize=20)

# Draw nodes
nx.draw_networkx_nodes(G, pos, nodelist=white_nodes, 
                       node_color='white', node_size=4200,
                       edgecolors='black', linewidths=2)
nx.draw_networkx_nodes(G, pos, nodelist=black_nodes, 
                       node_color='black', node_size=4200,
                       edgecolors='white', linewidths=2)

# Draw labels separately for white and black nodes
nx.draw_networkx_labels(G, pos, 
                       labels={n: G.nodes[n]['label'] for n in white_nodes},
                       font_size=16, font_weight='bold', font_color='black')
nx.draw_networkx_labels(G, pos, 
                       labels={n: G.nodes[n]['label'] for n in black_nodes},
                       font_size=16, font_weight='bold', font_color='white')

plt.title("Minimax Tree: White (○) vs Black (●)")
plt.axis('off')
plt.tight_layout()
plt.savefig("fig5.png")
plt.show()


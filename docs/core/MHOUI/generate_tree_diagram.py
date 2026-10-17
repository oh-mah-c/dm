import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set clean professional typography
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'

fig, ax = plt.subplots(figsize=(15.2, 9.0), dpi=300)
ax.set_xlim(-1, 15.5)
ax.set_ylim(-1.0, 9.0)
ax.axis('off')

# Title & Subtitle in formal academic style (no raw LaTeX commands)
ax.text(7.2, 8.6, 'Set-Enumeration Tree Traversal and Multi-Tier In-Process Pruning in MHOUI-Miner', 
        fontsize=13.5, fontweight='bold', ha='center', va='center', color='#000000')
ax.text(7.2, 8.15, r'Exact execution trace on benchmark database $\mathcal{D}_{\mathrm{toy}}$ ($\sigma = 3, \beta = 0.40, \mu = \$20$, Canonical Item Order: $d \prec b \prec a \prec c$)',
        fontsize=10.5, fontstyle='italic', ha='center', va='center', color='#333333')

def draw_node(x, y, text, subtext, status='normal', width=2.45, height=1.05):
    # status: 'root', 'mhoui', 'pruned_sup', 'pruned_dom', 'leap', 'normal'
    if status == 'root':
        fc = '#F5F5F5'
        ec = '#000000'
        ls = '-'
        lw = 1.6
        hatch = None
    elif status == 'mhoui':
        fc = '#EAEAEA'  # 10% solid grayscale for maximal patterns
        ec = '#000000'
        ls = '-'
        lw = 2.4        # Heavy bold border for MHOUI
        hatch = None
    elif status == 'pruned_sup':
        fc = '#FAFAFA'
        ec = '#444444'
        ls = '--'
        lw = 1.3
        hatch = '//'    # Clean light hatch
    elif status == 'pruned_dom':
        fc = '#FAFAFA'
        ec = '#444444'
        ls = '--'
        lw = 1.3
        hatch = '..'    # Dotted hatch
    elif status == 'leap':
        fc = '#F8F8F8'
        ec = '#000000'
        ls = '-'
        lw = 1.4
        hatch = None
    elif status == 'normal':
        fc = '#FFFFFF'
        ec = '#000000'
        ls = '-'
        lw = 1.3
        hatch = None

    box = patches.FancyBboxPatch((x - width/2, y - height/2), width, height,
                                boxstyle='square,pad=0.0',
                                facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, hatch=hatch)
    ax.add_patch(box)
    
    # Text background pill for hatched boxes to guarantee 100% readability
    if hatch:
        bbox_props = dict(boxstyle='square,pad=0.15', facecolor='#FFFFFF', edgecolor='none', alpha=0.92)
    else:
        bbox_props = None

    ax.text(x, y + 0.22, text, fontsize=9.8, fontweight='bold', ha='center', va='center', color='#000000', bbox=bbox_props)
    ax.text(x, y - 0.20, subtext, fontsize=8.0, ha='center', va='center', color='#111111', bbox=bbox_props)

def draw_edge(x1, y1, x2, y2, color='#000000', style='-', lw=1.3):
    ax.annotate('', xy=(x2, y2 + 0.525), xytext=(x1, y1 - 0.525),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw, linestyle=style))

# -------------------------------------------------------------
# ROOT NODE
# -------------------------------------------------------------
draw_node(7.0, 7.0, r'Root: $\emptyset$', r'$\mathcal{I}_{\mathrm{prom}} = \{d, b, a, c\}$' + '\n' + r'$TU(\mathcal{D}) = \$64$', 'root', width=2.9, height=1.1)

# Root-level pruned item e
draw_node(13.4, 7.0, r'Item $\{e\}$ (Pruned)', r'$\operatorname{supp}(e) = 1 < \sigma$' + '\n' + r'(Anti-monotone cutoff)', 'pruned_sup', width=2.6, height=1.1)
ax.annotate('', xy=(12.1, 7.0), xytext=(8.45, 7.0),
            arrowprops=dict(arrowstyle='->', color='#333333', lw=1.3, linestyle=':'))
ax.text(10.3, 7.28, r'Pre-pruning ($s < \sigma$)', fontsize=8.2, color='#333333', fontstyle='italic', ha='center')

# -------------------------------------------------------------
# LEVEL 1: d, b, a, c
# -------------------------------------------------------------
# Node d
draw_edge(7.0, 7.0, 1.8, 4.9)
draw_node(1.8, 4.9, r'Prefix $\{d\}$', r'$\operatorname{supp}=3, u=\$8$' + '\n' + r'$\operatorname{aocc}=0.3056 < \beta$', 'normal')

# Node b
draw_edge(7.0, 7.0, 5.5, 4.9)
draw_node(5.5, 4.9, r'Prefix $\{b\}$', r'$\operatorname{supp}=4, u=\$7$' + '\n' + r'$\operatorname{aocc}=0.2917 < \beta$', 'normal')

# Node a
draw_edge(7.0, 7.0, 9.6, 4.9)
draw_node(9.6, 4.9, r'Prefix $\{a\}$', r'$\operatorname{supp}=4, u=\$15$' + '\n' + r'$\operatorname{aocc}=0.2917 < \beta$', 'leap')

# Node c
draw_edge(7.0, 7.0, 13.4, 4.9)
draw_node(13.4, 4.9, r'Prefix $\{c\}$', r'$\operatorname{supp}=5, u=\$30, \operatorname{aocc}=0.30$' + '\n' + r'[Pruned: $\operatorname{aocc}<\beta$, Subsumed]', 'pruned_dom', width=2.6)

# -------------------------------------------------------------
# LEVEL 2: UNDER d
# -------------------------------------------------------------
# {d, b}
draw_edge(1.8, 4.9, 0.4, 2.8)
draw_node(0.4, 2.8, r'$\{d, b\}$', r'$\operatorname{supp}=2 < \sigma$' + '\n' + r'[Cut: Anti-monotone]', 'pruned_sup', width=1.8)

# {d, a}
draw_edge(1.8, 4.9, 2.3, 2.8)
draw_node(2.3, 2.8, r'$\{d, a\}$', r'$\operatorname{supp}=2 < \sigma$' + '\n' + r'[Cut: Anti-monotone]', 'pruned_sup', width=1.8)

# {d, c} -> MHOUI !
draw_edge(1.8, 4.9, 4.3, 2.8)
draw_node(4.3, 2.8, r'$\{d, c\}$', r'$\operatorname{supp}=3, u=\$23, \operatorname{aocc}=0.6111$' + '\n' + r'[Maximal Pattern $\mathcal{M}_1$]', 'mhoui', width=2.3)

# -------------------------------------------------------------
# LEVEL 2: UNDER b
# -------------------------------------------------------------
# {b, a} -> Galois Closure Leap to {a, b, c}
draw_edge(5.5, 4.9, 7.0, 2.8)
draw_node(7.0, 2.8, r'$\{b, a\}$', r'$\operatorname{supp}=3, u=\$15$' + '\n' + r'[Closure Leap Trigger]', 'leap', width=2.2)

# {b, c} -> Dominated by {a, b, c}
draw_edge(5.5, 4.9, 9.6, 2.8)
draw_node(9.6, 2.8, r'$\{b, c\}$', r'$\operatorname{supp}=4, u=\$32, \operatorname{aocc}=0.5833$' + '\n' + r'[Pruned: $\prec_{\mathrm{s}} \{a,b,c\}$]', 'pruned_dom', width=2.5)

# -------------------------------------------------------------
# GALOIS CLOSURE LEAP FROM {a} TO {a, c}
# -------------------------------------------------------------
ax.annotate('', xy=(12.8, 2.8 + 0.525), xytext=(10.3, 4.9 - 0.525),
            arrowprops=dict(arrowstyle='->', color='#000000', lw=2.2, linestyle='--'))
ax.text(12.3, 4.10, 'Galois Closure Leap (O(1))\n' + r'$\operatorname{supp}(\{a,c\}) = \operatorname{supp}(\{a\}) = 4$', 
        fontsize=8.5, fontweight='bold', color='#000000', ha='center', 
        bbox=dict(boxstyle='square,pad=0.2', facecolor='#FFFFFF', edgecolor='#000000', lw=0.9))

# {a, c} -> MHOUI !
draw_node(12.8, 2.8, r'$\{a, c\}$', r'$\operatorname{supp}=4, u=\$40, \operatorname{aocc}=0.5833$' + '\n' + r'[Maximal Pattern $\mathcal{M}_2$]', 'mhoui', width=2.4)

# -------------------------------------------------------------
# LEVEL 3: GALOIS LEAP FROM {b, a} TO {a, b, c}
# -------------------------------------------------------------
ax.annotate('', xy=(7.0, 0.7 + 0.525), xytext=(7.0, 2.8 - 0.525),
            arrowprops=dict(arrowstyle='->', color='#000000', lw=2.2, linestyle='--'))
ax.text(8.3, 1.75, 'Galois Closure Leap\n' + r'$\operatorname{supp}(\{a,b,c\}) = \operatorname{supp}(\{b,a\}) = 3$', 
        fontsize=8.3, fontweight='bold', color='#000000', ha='left', 
        bbox=dict(boxstyle='square,pad=0.2', facecolor='#FFFFFF', edgecolor='#000000', lw=0.9))

# {a, b, c} -> MHOUI !
draw_node(7.0, 0.7, r'$\{a, b, c\}$', r'$\operatorname{supp}=3, u=\$35, \operatorname{aocc}=0.8333$' + '\n' + r'[Maximal Pattern $\mathcal{M}_3$]', 'mhoui', width=2.4)

# -------------------------------------------------------------
# SCIENTIFIC LEGEND (Monochrome / Grayscale)
# -------------------------------------------------------------
box = patches.FancyBboxPatch((-0.8, -0.85), 6.1, 2.2, boxstyle='square,pad=0.0',
                             facecolor='#FFFFFF', edgecolor='#000000', linewidth=1.2)
ax.add_patch(box)
ax.text(-0.6, 1.05, 'Search Space Pruning Taxonomy (Monochrome):', fontsize=9.2, fontweight='bold', color='#000000')

def draw_legend_item(x, y, fc, ec, text, lw=1.2, ls='-', hatch=None):
    ax.add_patch(patches.Rectangle((x, y-0.13), 0.38, 0.26, facecolor=fc, edgecolor=ec, lw=lw, linestyle=ls, hatch=hatch))
    ax.text(x + 0.50, y, text, fontsize=8.0, va='center', color='#000000')

draw_legend_item(-0.6, 0.68, '#EAEAEA', '#000000', r'Maximal High-Occupancy Utility Itemset ($\mathcal{MHOUI}$)', lw=2.2)
draw_legend_item(-0.6, 0.32, '#FAFAFA', '#444444', r'Pruned: Support Anti-monotonicity ($\operatorname{supp} < \sigma$)', ls='--', hatch='//')
draw_legend_item(-0.6, -0.04, '#FAFAFA', '#444444', r'Pruned: Pareto Dominance ($X \prec_{\mathrm{s}} Y$)', ls='--', hatch='..')
draw_legend_item(-0.6, -0.40, '#FFFFFF', '#000000', r'Galois Closure Leap ($O(1)$ intermediate jump)', ls='--')
draw_legend_item(-0.6, -0.72, '#FFFFFF', '#000000', r'Candidate Prefix Exploration Node', lw=1.2)

plt.tight_layout()
plt.savefig('mhoui_search_tree.png', dpi=300, bbox_inches='tight')
print('Successfully generated clean academic monochrome mhoui_search_tree.png')

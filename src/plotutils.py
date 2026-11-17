from src.latexfig import LatexFigure
import numpy as np
from scipy.optimize import curve_fit

def plot_lines(df, xaxes=['SplitNorm'], yaxes=['Enthalpy'], width=1, AR=0.62):

    units = {
        'SplitNorm': '',
        'EnergyShift': 'Ry',
        'MeshCutoff': 'Ry',
        'kgrid': 'x, x, x'
    }
    N = len(xaxes)
    M = len(yaxes)

    lf = LatexFigure()
    fig, axes = lf.create(subplots=(M, N), sharex='col', sharey='row',
                          width=width, AR=AR, grid=True)

    def plot_df(ax, df, xaxis, yaxis):
        # Find the column that is not xaxis to group by
        cols = df.columns[0:2]
        # If there is a 'kgrid' column, we need to handle it differently
        if 'kgrid' in cols:
            # Convert in dataframe to int for plotting only internally in the function, we don't want to modify the original dataframe
            df = df.copy()
            df['kgrid'] = df['kgrid'].apply(lambda x: int(x.split(',')[0].strip('()')))
        group_col = [col for col in cols if col != xaxis][0]
        
        # Remove all rows where group_col is not equal to its maximum value
        df = df[df[group_col] == df[group_col].max()]
        # Sort from lowest to highest value of xaxis
        df = df.sort_values(by=xaxis)
        # Plot data
        x = np.array(df[xaxis])
        y = np.array(df[yaxis])

        # Subtract the first value of y
        y -= y[0]
        y = y*1000
        if group_col == 'kgrid':
            label = f'{group_col}=({df[group_col].iloc[0]},{df[group_col].iloc[0]},{df[group_col].iloc[0]})'
        else:
            label = f'{group_col}={df[group_col].iloc[0]} {units[group_col]}'
        ax.plot(x, y, marker='o', label=label)
        # Add legend
        ax.legend(loc='upper left')
    
    for i, xaxis in enumerate(xaxes):
        for j, yaxis in enumerate(yaxes):
            plot_df(axes[i+N*j], df, xaxis, yaxis)

    for i, xaxis in enumerate(xaxes):
        # Add x-label
        axes[i+N].set_xlabel(xaxis + f' ({units[xaxis]})')

    for j, yaxis in enumerate(yaxes):
        # Add y-label
        if yaxis == 'MaxForce':
            label = r'$\Delta$ ' + yaxis + r' (meV/Å)'
        else:
            label = r'$\Delta$ ' + yaxis + r' (meV)'
        axes[N*j].set_ylabel(label)
    
    return fig

def plot_contour(df, vals='Enthalpy', width=1, AR=0.62):

    units = {
        'SplitNorm': '',
        'EnergyShift': ' (Ry)'
    }
    
    cols = df.columns[0:2]
    Z = df.pivot(index=cols[0], columns=cols[1], values=vals)
    Z -= np.nanmin(Z.values)
    #Z = Z*1000

    X, Y = np.meshgrid(Z.columns.values, Z.index.values)
    # Plotting the log-likelihood contour
    lf = LatexFigure()
    fig, axes = lf.create(subplots=(1, 1), width=width, AR=AR, minor=False)
    ax = axes[0]
    CS = ax.contour(X,Y,Z, 100, cmap='viridis')
    if vals == 'TotalForce':
        label = r'$\Delta$ ' + vals + r' (eV/Å)'
    else:
        label = r'$\Delta$ ' + vals + r' (eV)'
    
    fig.colorbar(CS, ax=ax, label=label)
    # Add x- and y-labels
    ax.set_xlabel(cols[1] + units[cols[1]])
    ax.set_ylabel(cols[0] + units[cols[0]])

    ax.set_xticks(X[0,:], np.round(X[0,:], 3))
    ax.set_yticks(Y[:,0], np.round(Y[:,0], 3))

    return fig

# Double well potential
# ---------- Models ----------

def U_anharm(x, a, b):
    return -0.5 * a * x**2 + 0.25 * b * x**4


# ---------- Utilities ----------

def symmetrize_data(x, y):
    x_sym = np.concatenate((-x[::-1], x))
    y_sym = np.concatenate((y[::-1], y))
    return x_sym, y_sym


def extract_dw_properties(a, b):
    x0 = np.sqrt(a / b)
    barrier = -a**2 / (4 * b)
    return x0, barrier


def fit_double_well(x, y, fit_lim=10):
    x, y = symmetrize_data(x, y)

    # Shift minimum to zero reference
    y = y - y[np.argmin(np.abs(x))]

    popt, _ = curve_fit(
        U_anharm,
        x[y < fit_lim],
        y[y < fit_lim],
        bounds=([0, 0], [np.inf, np.inf])
    )

    a, b = popt
    xfit = np.linspace(min(x), max(x), 1000)
    yfit = U_anharm(xfit, a, b)

    x0, barrier = extract_dw_properties(a, b)

    return x, y, xfit, yfit, x0, barrier


def plot_frozen(results, labels, colors=None, lstyles=None, pstyles=None,
                width=1, AR=1.7, mirror=False, strained=False, skip_q=[], Natoms=1, fit_lims=10):

    def _ensure_list(obj):
        if isinstance(obj, list):
            return obj
        else:
            return [obj]
        
    # Ensure that results is a list of results dictionaries
    results = _ensure_list(results)

    if not isinstance(fit_lims, list):
        fit_lims = [fit_lims] * len(results)

    # Check the number of q-points in each results dictionary
    q_points = [set(res.keys()) for res in results]

    # Combine list of sets into a single set of all q-points
    qpoints = list(set.union(*q_points))

    # Sort list according to ['G', 'X', 'M', 'R']
    order = ['G', 'X', 'M', 'R']
    qpoints.sort(key=lambda q: order.index(q) if q in order else len(order) + ord(q[0]))

    # Remove any q-points that are in the skip_q list
    qpoints = [q for q in qpoints if q not in skip_q]

    # Create figure
    #qpoints = results[0].keys()
    N = len(qpoints)
    lf = LatexFigure()
    fig, axes = lf.create(width=width, subplots=(1, N), AR=AR, sharey=True, grid=True)

    # Define colors for the different results dictionaries (up to 6)
    if colors is None:
        colors = ["black", "blue", "red", "purple", "orange", "green"]
    if lstyles is None:
        lstyles = ['-', '-', '-', '-', '-', '-']
    if pstyles is None:
        pstyles = ['.', '.', '.', '.', '.', '.']

    max_x = {q: np.array([]) for q in qpoints}
    min_y = []

    for j, res in enumerate(results):

        for i, q in enumerate(qpoints):

            ax = axes[i]
            try:
                res_items = res[q].items()
            except KeyError:
                print(f"KeyError: q-point {q} not found in results dictionary {j}")
                continue

            for g_id, modes in res_items:
                for mode_id, data in modes.items():
                    df = data["df"]

                    x = df["Amplitude"].to_numpy()
                    y = df["Energy"].to_numpy() * 1000 / Natoms  # meV/atom

                    xdata, ydata, xfit, yfit, x0, barrier = fit_double_well(x, y, fit_lim=fit_lims[j])
                    print(f"Result {j}, q-point {q}, mode {g_id}/{mode_id}: x0 = {x0:.3f}, barrier = {barrier:.3f} meV/atom")

                    max_x[q] = np.append(max_x[q], np.max(xdata))
                    min_y.append(np.min(ydata))

                    if mirror:
                        ax.plot(xdata, ydata, pstyles[j], markersize=4, color=colors[j])
                        if g_id == 'mode_1' and mode_id == 'Q_1':
                            ax.plot(xfit, yfit, lstyles[j], lw=1, color=colors[j], label=labels[j])
                        else:
                            ax.plot(xfit, yfit, lstyles[j], lw=1, color=colors[j])
                    else:
                        ax.plot(xdata[xdata >= 0], ydata[xdata >= 0], pstyles[j], markersize=4, color=colors[j])
                        if g_id == 'mode_1' and mode_id == 'Q_1':
                            ax.plot(xfit[xfit >= 0], yfit[xfit >= 0], lstyles[j], lw=1, color=colors[j], label=labels[j])
                        else:
                            ax.plot(xfit[xfit >= 0], yfit[xfit >= 0], lstyles[j], lw=1, color=colors[j])

                    if q == 'G':
                        ax.set_title(f'$\Gamma$ point')
                    else:
                        if q == 'R' and strained:
                            ax.set_title(f'{q}/A point')
                        else:
                            ax.set_title(f'{q} point')

    import math

    my = math.floor(np.min(min_y))
    # If my is odd, make it even
    if my % 2 != 0:
        my -= 1

    if my < -5.9:
        dy = 2
    else:
        dy = 1

    ytickmarks = np.arange(my, 5, dy)

    axes[0].set_yticks(ytickmarks, ytickmarks.astype(str))
    axes[0].set_ylim(ytickmarks[0], ytickmarks[-1])
    axes[0].set_ylabel(r'$\Delta E$ (meV/atom)')

    for i, ax in enumerate(axes):

        mx = np.max(max_x[qpoints[i]])

        if mx < 2:
            dx = 0.5
        else:
            dx = 1

        xtickmarks = np.arange(0, mx+dx, dx)
        
        if mirror:
            # Take xtickmarks and mirror them to the negative side, excluding the zero
            xtickmarks = np.concatenate((-xtickmarks[::-1][:-1], xtickmarks))
            
        
        ax.set_xlabel(f'$Q_{i+1}$ (amu$^{{1/2}}$Å)')

        xticklabels = np.where(xtickmarks % 1 == 0,
                                xtickmarks.astype(int).astype(str),
                                xtickmarks.astype(str))
        ax.set_xticks(xtickmarks, xticklabels)
        ax.set_xlim(xtickmarks[0], xtickmarks[-1])

    #axes[0].legend()
    #plt.show()
    return fig

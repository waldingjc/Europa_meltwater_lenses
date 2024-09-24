import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy.stats import gaussian_kde
import seaborn as sns

def file_read(address, foldername, filename, compress_frac=False):

    tip_locs = []
    KI = []
    KII = []
    KIII = []
    G = []
    phi = []
    psi = []

    try:
        f = open(address + "/" + foldername + "/" + filename, "r")
    except FileNotFoundError:
        #fab data
        tip_locs.append([0, 0, 0])
        KI.append(0)
        KII.append(0)
        KIII.append(0)
        G.append(0)
        phi.append(0)
        psi.append(0)
        data = [KI, KII, KIII, G, phi, psi]
        print("####### WARNING: FILE READ ERROR #######")
        #return tip_locs, data
        return [], []
    
    f.readline()

    step_count = 0
    
    for line in f:
        temp = line
        temp = temp.split(" ")
        if temp[0] == "Step":
            step_count += 1
            continue
        if step_count < 2: continue
        temp[1] = 0
        temp = [float(x) for x in temp]
        tip_loc = [temp[4], temp[5], temp[6]]
        tip_locs.append(tip_loc)

        KI_temp = temp[7]
        if KI_temp < 0: KI_temp = 0
        elif KI_temp > 1e20: KI_temp = 0
        KI.append(KI_temp)

        KII_temp = temp[8]
        if KII_temp < 0: KII_temp *= -1
        elif KII_temp > 1e20: KII_temp = 0
        KII.append(KII_temp)

        KIII_temp = temp[9]
        if KIII_temp < 0: KIII_temp *= -1
        elif KIII_temp > 1e20: KIII_temp = 0
        KIII.append(KIII_temp)

        G_temp = temp[10]
        G.append(G_temp)

        phi_temp = temp[11]
        phi.append(phi_temp)

        psi_temp = temp[12]
        psi.append(psi_temp)

    data = [KI, KII, KIII, G, phi, psi]

    if data == [[], [], [], [], [], []]:
        tip_locs.append([0, 0, 0])
        KI.append(0)
        KII.append(0)
        KIII.append(0)
        G.append(0)
        phi.append(0)
        psi.append(0)
        data = [KI, KII, KIII, G, phi, psi]
        print("####### WARNING: FILE EMPTY ERROR #######")
        #return tip_locs, data
        return [], []

    if compress_frac:
        data_short = []
        for data_type in data:
            divisor = len(data_type)
            total = 0
            for datum in data_type:
                if datum < 1e20:
                    total += datum
                else:
                    divisor -= 1
            total /= divisor
            data_short.append([total])

        x_total = 0
        y_total = 0
        z_total = 0
        dat_length = len(tip_locs)
        for i in np.arange(dat_length):
            x_total += tip_locs[i][0]
            y_total += tip_locs[i][1]
            z_total += tip_locs[i][2]
        x_total /= dat_length
        y_total /= dat_length
        z_total /= dat_length

        avg_loc = [x_total, y_total, z_total]#[y_total, z_total]

        tip_locs = [avg_loc]
        data = data_short
    
    return tip_locs, data

def file_read_vector(address, foldername, filename, compress_frac=False):
    tip_locs = []
    phi = []
    psi = []

    try:
        f = open(address + "/" + foldername + "/" + filename, "r")
    except FileNotFoundError:
        #fab data
        tip_locs.append([0, 0, 0])
        phi.append(0)
        psi.append(0)
        print("####### WARNING: FILE READ ERROR (VECTOR) #######")
        #return tip_locs, phi, psi
        return [], 0, 0
    
    f.readline()

    step_count = 0

    longest_throw = 0

    read_pair = True
    
    for line in f:
        temp = line
        temp = temp.split(" ")
        if temp[0] == ";Printing":
            step_count += 1
            continue
        if step_count < 2: continue
        if temp[0] != "_Line":
            if temp[0] != "_Point": continue
        if not read_pair: 
            read_pair = True
            continue

        loc1 = temp[1]
        loc2 = temp[2]

        if loc2 == "_Enter;point" or loc2 == "_Enter" or loc2 == "_Enter\n":
            loc1 = loc1.split(",")
            for i in np.arange(len(loc1)):
                loc1[i] = float(loc1[i])
            loc1 = np.array(loc1)
            tip_locs.append(loc1)
            phi.append(0)
            psi.append(0)
            read_pair = False
            continue

        loc1 = loc1.split(",")
        loc2 = loc2.split(",")

        for i in np.arange(len(loc1)):
            loc1[i] = float(loc1[i])
            loc2[i] = float(loc2[i])

        loc1 = np.array(loc1)
        loc2 = np.array(loc2)

        vec = loc2 - loc1

        phi_temp = np.arctan(vec[1] / vec[0])

        hypot = np.sqrt(vec[0]**2 + vec[1]**2)
        throw = np.sqrt(hypot**2 + vec[2])
        #print(throw)
        psi_temp = np.arctan(vec[2] / hypot)
        tip_locs.append(loc1)

        phi.append(phi_temp * 180 / np.pi)
        psi.append(psi_temp * 180 / np.pi)

        if throw > longest_throw: #very inefficient, improve if successful
            longest_throw = throw
            for i in np.arange(len(phi) - 1):
                phi[i] = phi[-1]
                psi[i] = psi[-1]

        read_pair = False

    if phi == []:
        #fab data
        tip_locs.append([0, 0, 0])
        phi.append(0)
        psi.append(0)
        print("####### WARNING: FILE EMPTY ERROR (VECTOR) #######")
        #return tip_locs, phi, psi
        return [], 0, 0

    if compress_frac:
        phi_total = 0
        psi_total = 0
        for i in np.arange(len(phi)):
            phi_total += phi[i]
            psi_total += psi[i]
        phi_total /= len(phi)
        psi_total /= len(psi)

        phi = [phi_total]
        psi = [psi_total]

        x_total = 0
        y_total = 0
        z_total = 0
        dat_length = len(tip_locs)
        for i in np.arange(dat_length):
            x_total += tip_locs[i][0]
            y_total += tip_locs[i][1]
            z_total += tip_locs[i][2]
        x_total /= dat_length
        y_total /= dat_length
        z_total /= dat_length

        avg_loc = [x_total, y_total, z_total]#[y_total, z_total]

        tip_locs = [avg_loc]


    return tip_locs, phi, psi

def file_read_vector_norm(address, foldername, filename):
    tip_locs = []
    phi = []
    psi = []

    try:
        f = open(address + "/" + foldername + "/" + filename, "r")
    except FileNotFoundError:
        #fab data
        tip_locs.append([0, 0, 0])
        phi.append(0)
        psi.append(0)
        print("####### WARNING: FILE READ ERROR (VECTOR) #######")
        #return tip_locs, phi, psi
        return [], 0, 0, 0, 0, []
    
    f.readline()

    step_count = 0

    longest_throw = 0

    shortest_throw = 1e10

    read_pair = True

    vec_long = [0, 0, 0]
    vec_short = [0, 0, 0]
    loc_long = [0, 0, 0]
    loc_short = [0, 0, 0]

    finished = False
    
    for line in f:
        temp = line
        temp = temp.split(" ")
        if temp[0] == ";Printing":
            step_count += 1
            continue
        if step_count < 2: continue
        if temp[0] != "_Line":
            if temp[0] != "_Point": continue
        if not read_pair: 
            read_pair = True
            continue
        #if temp[0] == ";Fracture3D::ComputePropagationVectors" and step_count != 0:
            #finished = True
            #continue
        #if finished: continue

        loc1 = temp[1]
        loc2 = temp[2]

        if loc2 == "_Enter;point" or loc2 == "_Enter" or loc2 == "_Enter\n":
            loc1 = loc1.split(",")
            for i in np.arange(len(loc1)):
                loc1[i] = float(loc1[i])
            loc1 = np.array(loc1)
            tip_locs.append(loc1)
            phi.append(0)
            psi.append(0)
            read_pair = False
            continue

        loc1 = loc1.split(",")
        loc2 = loc2.split(",")

        for i in np.arange(len(loc1)):
            loc1[i] = float(loc1[i])
            loc2[i] = float(loc2[i])

        loc1 = np.array(loc1)
        loc2 = np.array(loc2)

        vec = loc2 - loc1

        hypot = np.sqrt(vec[0]**2 + vec[1]**2)
        throw = np.sqrt(hypot**2 + vec[2])
        #print(throw)
        tip_locs.append(loc1)

        if throw > longest_throw:
            longest_throw = throw
            vec_long = vec
            loc_long = loc2
        elif throw < shortest_throw:
            shortest_throw = throw
            vec_short = vec   
            loc_short = loc2         

        read_pair = False

    if np.array_equal(vec_long, vec_short):
        print("##### ERROR DEGENERATE AXES #####")
        return [], 0, 0, 0, 0, []
    
    norm = np.cross(vec_long, vec_short)

    phi_temp = np.arctan(vec_long[1] / vec_long[0])
    hypot = np.sqrt(vec_long[0]**2 + vec_long[1]**2)
    psi_temp = np.arctan(vec_long[2] / hypot)

    phi = phi_temp * 180 / np.pi
    psi = psi_temp * 180 / np.pi

    x_total = 0
    y_total = 0
    z_total = 0
    dat_length = len(tip_locs)
    if dat_length == 0: return [], [], []
    for i in np.arange(dat_length):
        x_total += tip_locs[i][0]
        y_total += tip_locs[i][1]
        z_total += tip_locs[i][2]
    x_total /= dat_length
    y_total /= dat_length
    z_total /= dat_length

    avg_loc = np.array([x_total, y_total, z_total])

    norm = np.cross(loc_long - avg_loc, loc_short - avg_loc)

    phi_temp = np.arctan(norm[1] / norm[0])
    hypot = np.sqrt(norm[0]**2 + norm[1]**2)
    psi_temp = np.arctan(norm[2] / hypot)

    frac_phi = phi_temp * 180 / np.pi
    frac_psi = psi_temp * 180 / np.pi

    return [avg_loc], [phi], [psi], [frac_phi], [frac_psi], norm

def plotting(data_x, data_y, title, title_x, title_y): 
    y_max = data_y[0]
    y_min = data_y[0]
    for datum in data_y:
        if datum > y_max: y_max = datum
        elif datum < y_min: y_min = datum
    fig, ax = plt.subplots()
    ax.yaxis.major.formatter._useMathText = True
    ax.set_xlim([0, 45e3])
    #ax.set_ylim([0, 8e7])
    ax.set_ylim([y_min, y_max])
    sc = plt.scatter(data_x, data_y)
    plt.title(title)
    plt.xlabel(title_x)
    plt.ylabel(title_y)
    #plt.show()

def plotting_line(data_x, data_y, title, title_x, title_y, labels):    
    fig, ax = plt.subplots()
    ax.yaxis.major.formatter._useMathText = True
    for i in range(len(data_x)):
        sc = plt.plot(data_x[i], data_y[i], label=labels[i])
    plt.title(title)
    plt.xlabel(title_x)
    plt.ylabel(title_y)
    #plt.legend()
    #plt.show()

def plotting_line_long(data_x, data_y, title, title_x, title_y, labels, data_y_supplement=[], savename=0, diff=False, factor=1):    
    fig, ax = plt.subplots()
    min = np.array(data_y[0])
    max = np.array(data_y[2])
    if np.min(min) >= 0:
        ax.set_ylim(ymin=0)
        ax.set_ylim(ymax=np.max(max)* 1.1 / 1e6)
    else:
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.set_ylim(ymax=90)
        ax.set_ylim(ymin=-90)
    ax.set_xlim([0, 180])
    ax.yaxis.major.formatter._useMathText = True
    colors = plt.cm.cividis(np.linspace(0,1,len(data_y)))
    colors[1] = [112/255, 41/255, 99/255, 1]
    if data_y_supplement != []:
        for i in range(len(data_y_supplement)):
            if i % 4 == 0:
                sc = plt.plot(data_x, np.array(data_y_supplement[i]) / factor, alpha=0.2, color="k")
    for i in range(len(data_y)):
        sc = plt.plot(data_x, np.array(data_y[i]) / factor, label=labels[i], color=colors[i])
    if diff: sc = plt.plot(data_x, max - min, label="Difference")
    plt.title(title)
    plt.xlabel(title_x)
    plt.ylabel(title_y)
    plt.legend()
    #plt.show()
    
    if savename != 0:
        plt.savefig(savename + "_nolens.png", bbox_inches='tight')
        print("SAVED: " + savename)

def plotting_heatmap(tip_locs, data_z, title, title_x, title_y, title_z, savename=0):
    x = np.array([loc[0] for loc in tip_locs])
    y = np.array([loc[1] for loc in tip_locs])
    z = np.array(data_z)

    x /= 1000
    y /= 1000
    z_shifted = z - np.min(z) + 1

    assert len(x) == len(y) == len(z)
    
    ax_min = -50
    ax_max = 50
    ticksize = 10

    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, c=z, s=15000, cmap='cividis')
    plt.colorbar(label=title_z)
    #sns.kdeplot(x=x, y=y, weights=z_shifted, cmap='cividis', fill=True, thresh=0, levels=500)
    #sns.kdeplot(x=x, y=y, weights=z_shifted, fill=False, thresh=0, levels=20, color=(0, 0, 0, 0.4), linewidths=0.4)
    plt.xlim(ax_min, ax_max)
    plt.ylim(ax_min, ax_max)
    plt.xticks(np.arange(ax_min, ax_max+ticksize, ticksize))
    plt.yticks(np.arange(ax_min, ax_max+ticksize, ticksize))
    circle = plt.Circle((0, 0), 43, edgecolor='red', facecolor='none', linewidth=1)
    plt.gca().add_patch(circle)
    plt.gca().set_aspect('equal')

    plt.xlabel(title_x + " (km)")
    plt.ylabel(title_y + " (km)")
    plt.title(title + " || kde")
    #plt.show()

    if savename != 0:
        plt.savefig(savename + ".png", bbox_inches='tight')
        print("SAVED: " + savename)

def plotting_heatmap_2(tip_locs, data_z, title, title_x, title_y, title_z):

    x = np.array([loc[0] for loc in tip_locs])
    y = np.array([loc[1] for loc in tip_locs])
    z = np.array(data_z)

    x /= 1000
    y /= 1000
    z_shifted = z - np.min(z) + 1

    assert len(x) == len(y) == len(z)
    
    ax_min = -50
    ax_max = 50
    ticksize = 10

    x = np.round(x, 1)
    y = np.round(y, 1)

    unique_x = np.unique(x)
    unique_y = np.unique(y)

    # Reshape z into a 2D array based on unique x and y values
    z_grid = np.zeros((len(unique_y), len(unique_x)))

    for i in range(len(z)):
        # Find indices in the 2D grid
        x_index = np.where(unique_x == x[i])[0][0]
        y_index = np.where(unique_y == y[i])[0][0]
        z_grid[y_index, x_index] = z[i]

    # Plot the heatmap
    plt.figure(figsize=(8, 8))
    plt.imshow(z_grid, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='cividis', aspect='auto')
    plt.colorbar(label=title_z)

    plt.xlim(ax_min, ax_max)
    plt.ylim(ax_min, ax_max)
    plt.xticks(np.arange(ax_min, ax_max+ticksize, ticksize))
    plt.yticks(np.arange(ax_min, ax_max+ticksize, ticksize))
    circle = plt.Circle((0, 0), 43, edgecolor='red', facecolor='none', linewidth=1)
    plt.gca().add_patch(circle)
    plt.gca().set_aspect('equal')

    plt.xlabel(title_x + " (km)")
    plt.ylabel(title_y + " (km)")
    plt.title(title)

    #plt.scatter(x, y, c=z, s=100, cmap='cividis')
    #sns.kdeplot(x=x, y=y, weights=z_shifted, cmap='cividis', fill=True, thresh=0, levels=500)
    #sns.kdeplot(x=x, y=y, weights=z_shifted, fill=False, thresh=0, levels=20, color=(0, 0, 0, 0.4), linewidths=0.4)
    #plt.show()

def plotting_heatmad(tip_locs, data_z, title, title_x, title_y, title_z, savename=0, interp="cubic", refine=200j):

    x = np.array([loc[0] for loc in tip_locs])
    y = np.array([loc[1] for loc in tip_locs])
    z = np.array(data_z)

    x /= 1000
    y /= 1000
    z_shifted = z - np.min(z) + 1

    assert len(x) == len(y) == len(z)
    
    ax_min = -45
    ax_max = 45
    ticksize = 10

    x = np.round(x, 1)
    y = np.round(y, 1)

    # Define a fine grid
    grid_x, grid_y = np.mgrid[x.min():x.max():refine, y.min():y.max():refine]

    # Interpolate the data onto the fine grid
    z_interpolated = griddata((x, y), z, (grid_x, grid_y), method=interp)

    # Plot the interpolated heatmap
    # Plot the heatmap
    plt.figure(figsize=(8, 8))
    plt.imshow(z_interpolated.T, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='cividis', aspect='auto')#, vmin=0, vmax=25)
    plt.colorbar(label=title_z)

    contours = plt.contour(grid_x, grid_y, z_interpolated, colors='black', levels=20, linewidths=0.4, alpha=0.4)
    #plt.clabel(contours, inline=True, fontsize=8, fmt='%.2f')

    plt.xlim(ax_min, ax_max)
    plt.ylim(ax_min, ax_max)
    plt.xticks(np.arange(ax_min, ax_max+ticksize, ticksize))
    plt.yticks(np.arange(ax_min, ax_max+ticksize, ticksize))
    circle = plt.Circle((0, 0), 43, edgecolor='red', facecolor='none', linewidth=1)
    #plt.gca().add_patch(circle)
    plt.gca().set_aspect('equal')

    plt.xlabel(title_x + " (km)")
    plt.ylabel(title_y + " (km)")
    plt.title(title)

    if savename != 0:
        plt.savefig(savename + ".png", bbox_inches='tight')
        print("SAVED: " + savename)

    #plt.scatter(x, y, c=z, s=100, cmap='cividis')
    #sns.kdeplot(x=x, y=y, weights=z_shifted, cmap='cividis', fill=True, thresh=0, levels=500)
    #sns.kdeplot(x=x, y=y, weights=z_shifted, fill=False, thresh=0, levels=20, color=(0, 0, 0, 0.4), linewidths=0.4)
    #plt.show()

def plotting_quiver(tip_locs, data_z, title, title_x, title_y, savename=0):

    x = np.array([loc[0] for loc in tip_locs])
    y = np.array([loc[1] for loc in tip_locs])
    z = np.array(data_z)

    x /= 1000
    y /= 1000

    assert len(x) == len(y) == len(z)
    
    ax_min = -45
    ax_max = 45
    ticksize = 10

    x = np.round(x, 1)
    y = np.round(y, 1)
    plt.figure(figsize=(6, 6))
    u = np.cos(np.deg2rad(z)) * 5  # X components
    v = np.sin(np.deg2rad(z)) * 5  # Y components
    plt.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=1, color='k')#, headlength=0, headaxislength=0, headwidth=0)

    plt.grid()

    plt.xlim(ax_min, ax_max)
    plt.ylim(ax_min, ax_max)
    plt.xticks(np.arange(ax_min, ax_max+ticksize, ticksize))
    plt.yticks(np.arange(ax_min, ax_max+ticksize, ticksize))
    circle = plt.Circle((0, 0), 43, edgecolor='red', facecolor='none', linewidth=1)
    plt.gca().add_patch(circle)
    plt.gca().set_aspect('equal')

    plt.xlabel(title_x + " (km)")
    plt.ylabel(title_y + " (km)")
    plt.title(title)

    if savename != 0:
        plt.savefig(savename + "_2.png", bbox_inches='tight')
        print("SAVED: " + savename)

def plotting_quiver_vec(tip_locs, data_z, title, title_x, title_y, savename=0):

    x = np.array([loc[0] for loc in tip_locs])
    y = np.array([loc[1] for loc in tip_locs])

    x /= 1000
    y /= 1000

    assert len(x) == len(y) == len(data_z)
    
    ax_min = -45
    ax_max = 45
    ticksize = 10

    x = np.round(x, 1)
    y = np.round(y, 1)
    plt.figure(figsize=(6, 6))
    u = [vec[0] for vec in data_z]  # X components
    v = [vec[1] for vec in data_z]  # Y components
    plt.quiver(x, y, u, v, angles='xy', scale_units='xy', color='r')

    plt.grid()

    plt.xlim(ax_min, ax_max)
    plt.ylim(ax_min, ax_max)
    plt.xticks(np.arange(ax_min, ax_max+ticksize, ticksize))
    plt.yticks(np.arange(ax_min, ax_max+ticksize, ticksize))
    circle = plt.Circle((0, 0), 43, edgecolor='red', facecolor='none', linewidth=1)
    plt.gca().add_patch(circle)
    plt.gca().set_aspect('equal')

    plt.xlabel(title_x + " (km)")
    plt.ylabel(title_y + " (km)")
    plt.title(title)

    if savename != 0:
        plt.savefig(savename + ".png", bbox_inches='tight')
        print("SAVED: " + savename)

#address = "C:/Europa_Runs/runs/quant_fix-dat_31.07.2024_time_12.21"
#address = "E:/Europa_Archive/quant_test-1"
#address = "E:/Europa_Archive/quant_heatmap-dat_10.07.2024_time_.0.09"
#address = "E:/Europa_Archive/quant_heatmap-fix-dat"
#address = "E:/Europa_Archive/quant_deep_run"
#address = "E:/Europa_Archive/quant_high_detail_long"
#address = "E:/Europa_Archive/quant_fix_nolens-dat_EXTRACT"
address = "E:/Europa_Archive/quant_fix_nolens-longi_EXTRACT"

filename = "fracture_sif_data_raw.txt"
filename_vector = "simulation_propagation_vectors.txt"

save_folder = "C:/Europa_Runs/cuts/nolens_longi/"

folder_longitude = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180]
#folder_longitude = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
#folder_longitude = [0, 30, 60, 90, 120, 150]
#folder_longitude = [0, 60, 120]
#folder_longitude = [0]
folder_x_number = [0]
folder_y_number = [0]
#folder_x_number = [0, 1, 2, 3]
#folder_y_number = [0, 1, 2, 3]
#folder_x_number = [0, 3, 6]
#folder_y_number = [0, 3, 6]
#folder_x_number = [0, 1, 2, 3, 4, 5, 6]
#folder_y_number = [0, 1, 2, 3, 4, 5, 6]
#folder_x_number = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
#folder_y_number = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
orientation_codes = ["hori", "vert", "diag"]
#orientation_codes = ["hori"]
#orientation_codes = ["vert"]
#orientation_codes = ["diag"]

tip_locs_long = []
data_long = []
vec_data_long = []

for longitude in folder_longitude:
    tip_locs = []
    data = [[], [], [], [], [], [], [], []]
    vec_data = []
    for x_index in folder_x_number:
        for y_index in folder_y_number:
            for orientation in orientation_codes:
                foldername = "run_" + str(longitude) + "_" + str(x_index) + "-" + str(y_index) + "_" + str(orientation)
                #tip_locs_temp, data_temp = file_read(address, foldername, filename, False)
                tip_locs_temp, data_temp = file_read(address, foldername, filename, True)
                #tip_locs_angle_temp, phi_temp, psi_temp = file_read_vector(address, foldername, filename_vector, True)
                tip_locs_angle_temp, phi_temp, psi_temp, phi_frac_temp, psi_frac_temp, vec_temp = file_read_vector_norm(address, foldername, filename_vector)
                if tip_locs_temp == [] or tip_locs_angle_temp == []:
                    if tip_locs == []:
                        tip_locs.append(tip_locs_long[-1][0])
                        for i in np.arange(len(data)):
                            data[i].append(data_long[-1][i][-1])
                        vec_data.append(vec_data_long[-1][-1])
                    tip_locs.append(tip_locs[-1])
                    for i in np.arange(len(data)):
                        data[i].append(data[i][-1])
                    vec_data.append(vec_data[-1])
                    print(foldername + " || failure")
                    continue
                vec_data.append(vec_temp)
                tip_locs += tip_locs_temp
                data_temp[-2] = phi_temp
                data_temp[-1] = psi_temp
                data_temp.append(phi_frac_temp)
                data_temp.append(psi_frac_temp)
                for i in np.arange(len(data)):
                    data[i] += data_temp[i]

                print(foldername + " || complete")
    tip_locs_long.append(tip_locs)
    data_long.append(np.array(data))
    vec_data_long.append(vec_data)

rad_vals_long = []
for tip_locs in tip_locs_long:
    rad_vals = []
    for loc in tip_locs:
        rad = np.sqrt(loc[0]**2 + loc[1]**2)
        #rad_vals.append(rad)
        rad_vals.append(loc[0])
    rad_vals_long.append(rad_vals)

KI_long = [[], [], []]
KII_long = [[], [], []]
KIII_long = [[], [], []]
KV_long = [[], [], []]
G_long = [[], [], []]
phi_long = [[], [], []]
psi_long = [[], [], []]
phi_frac_long = [[], [], []]
psi_frac_long = [[], [], []]

data_supp = [[], [], [], [], [], [], [], []]
for datum in np.arange(len(data_supp)):
    for i in np.arange(len(data_long[0][0])):
        #if i % 4 != 0: continue
        data_supp_temp = []
        for j in np.arange(len(data_long)):
            data_supp_temp.append(data_long[j][datum][i])
        data_supp[datum].append(data_supp_temp)

for i in np.arange(len(folder_longitude)):
    KI = data_long[i][0]
    KII = data_long[i][1]
    KIII = data_long[i][2]
    KV = []
    G = data_long[i][3]
    phi = data_long[i][4]
    psi = data_long[i][5]
    phi_frac = data_long[i][6]
    psi_frac = data_long[i][7]
    vec = vec_data_long[i]
    alpha_1 = 1 / 1.3268
    alpha_2 = 1 / 0.6297
    for j in np.arange(len(KI)):
        alpha_1_temp = 4 * (alpha_1 * KII[j])**2
        alpha_2_temp = 4 * (alpha_2 * KIII[j])**2

        KV_temp = 0.5 * KI[j] + 0.5 * np.sqrt(KI[j]**2 + alpha_1_temp + alpha_2_temp)

        KV.append(KV_temp)


    KI_dat = [KI[0], 0, KI[0]]
    KII_dat = [KII[0], 0, KII[0]]
    KIII_dat = [KIII[0], 0, KIII[0]]
    KV_dat = [KV[0], 0, KV[0]]
    G_dat = [G[0], 0, G[0]]
    phi_dat = [phi[0], 0, phi[0]]
    psi_dat = [psi[0], 0, psi[0]]
    phi_frac_dat = [phi_frac[0], 0, phi_frac[0]]
    psi_frac_dat = [psi_frac[0], 0, psi_frac[0]]

    for datum in np.arange(len(KI)):
        KI_dat[1] += KI[datum]
        if KI[datum] < KI_dat[0]: KI_dat[0] = KI[datum]
        if KI[datum] > KI_dat[2]: KI_dat[2] = KI[datum]
        KII_dat[1] += KII[datum]
        if KII[datum] < KII_dat[0]: KII_dat[0] = KII[datum]
        if KII[datum] > KII_dat[2]: KII_dat[2] = KII[datum]
        KIII_dat[1] += KIII[datum]
        if KIII[datum] < KIII_dat[0]: KIII_dat[0] = KIII[datum]
        if KIII[datum] > KIII_dat[2]: KIII_dat[2] = KIII[datum]
        KV_dat[1] += KV[datum]
        if KV[datum] < KV_dat[0]: KV_dat[0] = KV[datum]
        if KV[datum] > KV_dat[2]: KV_dat[2] = KV[datum]
        G_dat[1] += G[datum]
        if G[datum] < G_dat[0]: G_dat[0] = G[datum]
        if G[datum] > G_dat[2]: G_dat[2] = G[datum]
        phi_dat[1] += phi[datum]
        if phi[datum] < phi_dat[0]: phi_dat[0] = phi[datum]
        if phi[datum] > phi_dat[2]: phi_dat[2] = phi[datum]
        psi_dat[1] += psi[datum]
        if psi[datum] < psi_dat[0]: psi_dat[0] = psi[datum]
        if psi[datum] > psi_dat[2]: psi_dat[2] = psi[datum]
        phi_frac_dat[1] += phi_frac[datum]
        if phi_frac[datum] < phi_frac_dat[0]: phi_frac_dat[0] = phi_frac[datum]
        if phi_frac[datum] > phi_frac_dat[2]: phi_frac_dat[2] = phi_frac[datum]
        psi_frac_dat[1] += psi_frac[datum]
        if psi_frac[datum] < psi_frac_dat[0]: psi_frac_dat[0] = psi_frac[datum]
        if psi_frac[datum] > psi_frac_dat[2]: psi_frac_dat[2] = psi_frac[datum]

    KI_dat[1] /= len(KI)
    KII_dat[1] /= len(KII)
    KIII_dat[1] /= len(KIII)
    KV_dat[1] /= len(KV)
    G_dat[1] /= len(G)
    phi_dat[1] /= len(phi)
    psi_dat[1] /= len(psi)
    phi_frac_dat[1] /= len(phi)
    psi_frac_dat[1] /= len(psi)

    for index in [0, 1, 2]:
        KI_long[index].append(KI_dat[index])
        KII_long[index].append(KII_dat[index])
        KIII_long[index].append(KIII_dat[index])
        KV_long[index].append(KV_dat[index])
        G_long[index].append(G_dat[index])
        phi_long[index].append(phi_dat[index])
        psi_long[index].append(psi_dat[index])
        phi_frac_long[index].append(phi_frac_dat[index])
        psi_frac_long[index].append(psi_frac_dat[index])

    KV = np.array(KV)

    if len(orientation_codes) == 1:
        if orientation_codes[0] == "hori":
            orientation_name = "horizontal"
        elif orientation_codes[0] == "vert":
            orientation_name = "vertical"
        elif orientation_codes[0] == "diag":
            orientation_name = "diagonal"
    else:
        orientation_name = "combined"

    savename = "_" + str(folder_longitude[i]) + "_" + orientation_name

    #plotting(rad_vals_long[i], KI,     "Longitude: " + str(folder_longitude[i]) + " || Radius v KI",       " Radius", "KI")
    #plotting(rad_vals_long[i], KII,    "Longitude: " + str(folder_longitude[i]) + " || Radius v KII",      " Radius", "KII")
    #plotting(rad_vals_long[i], KIII,   "Longitude: " + str(folder_longitude[i]) + " || Radius v KIII",     " Radius", "KIII")
    #plotting(rad_vals_long[i], KV,     "Longitude: " + str(folder_longitude[i]) + " || Radius v KV",       " Radius", "KV")
    #plotting(rad_vals_long[i], G,      "Longitude: " + str(folder_longitude[i]) + " || Radius v G",        " Radius", "G")
    #plotting(rad_vals_long[i], phi,    "Longitude: " + str(folder_longitude[i]) + " || Radius v phi",      " Radius", "phi")
    #plotting(rad_vals_long[i], psi,    "Longitude: " + str(folder_longitude[i]) + " || Radius v psi",      " Radius", "psi")

    #plotting_heatmap(tip_locs_long[i], KI / 1e6,     "KI || Longitude: " + str(folder_longitude[i]) + " || Orientation: " + orientation_name, "x", "y", "KI (MPa . m^0.5)",     save_folder + "KI" + savename)
    #plotting_heatmap(tip_locs_long[i], KII / 1e6,   "KII || Longitude: " + str(folder_longitude[i]) + " || Orientation: " + orientation_name, "x", "y", "KII (MPa . m^0.5)",    save_folder + "KII" + savename)
    #plotting_heatmap(tip_locs_long[i], KIII / 1e6, "KIII || Longitude: " + str(folder_longitude[i]) + " || Orientation: " + orientation_name, "x", "y", "KIII (MPa . m^0.5)",   save_folder + "KIII" + savename)
    #plotting_heatmap(tip_locs_long[i], KV / 1e6,     "KV || Longitude: " + str(folder_longitude[i]) + " || Orientation: " + orientation_name, "x", "y", "KV (MPa . m^0.5)",     save_folder + "KV" + savename)
    #plotting_heatmap(tip_locs_long[i], G,             "G || Longitude: " + str(folder_longitude[i]) + " || Orientation: " + orientation_name, "x", "y", "G",                    save_folder + "G" + savename)
    #plotting_heatmap(tip_locs_long[i], phi,         "phi || Longitude: " + str(folder_longitude[i]) + " || Orientation: " + orientation_name, "x", "y", "phi (deg)",            save_folder + "phi" + savename)
    #plotting_heatmap(tip_locs_long[i], psi,         "psi || Longitude: " + str(folder_longitude[i]) + " || Orientation: " + orientation_name, "x", "y", "psi (deg)",            save_folder + "psi" + savename)

    #plotting_heatmad(tip_locs_long[i], KI / 1e6,     "KI || Longitude: " + str(folder_longitude[i]) + " || Orientation: " + orientation_name, "x", "y", "KI (MPa . m^0.5)",     save_folder + "KI" + savename)
    #plotting_heatmad(tip_locs_long[i], KII / 1e6,   "KII || Longitude: " + str(folder_longitude[i]) + " || Orientation: " + orientation_name, "x", "y", "KII (MPa . m^0.5)",    save_folder + "KII" + savename)
    #plotting_heatmad(tip_locs_long[i], KIII / 1e6, "KIII || Longitude: " + str(folder_longitude[i]) + " || Orientation: " + orientation_name, "x", "y", "KIII (MPa . m^0.5)",   save_folder + "KIII" + savename)
    #plotting_heatmad(tip_locs_long[i], KV / 1e6,     "KV || Longitude: " + str(folder_longitude[i]) + " || Orientation: " + orientation_name, "x", "y", "KV (MPa . m^0.5)",     save_folder + "KV" + savename)
    #plotting_heatmad(tip_locs_long[i], G,             "G || Longitude: " + str(folder_longitude[i]) + " || Orientation: " + orientation_name, "x", "y", "G",                    save_folder + "G" + savename)
    #plotting_heatmad(tip_locs_long[i], phi,         "phi || Longitude: " + str(folder_longitude[i]) + " || Orientation: " + orientation_name, "x", "y", "phi (deg)")#,            save_folder + "phi" + savename)
    #plotting_heatmad(tip_locs_long[i], psi,         "psi || Longitude: " + str(folder_longitude[i]) + " || Orientation: " + orientation_name, "x", "y", "psi (deg)")#,            save_folder + "psi" + savename)
    #plotting_heatmad(tip_locs_long[i], phi_frac,  "phi_f || Longitude: " + str(folder_longitude[i]) + " || Orientation: " + orientation_name, "x", "y", "phi (deg)")#,            save_folder + "phi" + savename)
    #plotting_heatmad(tip_locs_long[i], psi_frac,  "psi_f || Longitude: " + str(folder_longitude[i]) + " || Orientation: " + orientation_name, "x", "y", "psi (deg)")#,            save_folder + "psi" + savename)
    
    #plotting_quiver(tip_locs_long[i], phi,        "phi || Longitude: " + str(folder_longitude[i]) + " || Orientation: " + orientation_name, "x", "y", save_folder + "phi" + savename)
    #plotting_quiver(tip_locs_long[i], psi,        "psi || Longitude: " + str(folder_longitude[i]) + " || Orientation: " + orientation_name, "x", "y", save_folder + "psi" + savename)
    #plotting_quiver(tip_locs_long[i], phi_frac, "phi_f || Longitude: " + str(folder_longitude[i]) + " || Orientation: " + orientation_name, "x", "y", save_folder + "phi_frac" + savename)
    #plotting_quiver(tip_locs_long[i], psi_frac, "psi_f || Longitude: " + str(folder_longitude[i]) + " || Orientation: " + orientation_name, "x", "y", save_folder + "psi_frac" + savename)
    #plotting_quiver_vec(tip_locs_long[i], vec,  "vec   || Longitude: " + str(folder_longitude[i]) + " || Orientation: " + orientation_name, "x", "y", save_folder + "vec" + savename)
#plt.show()

savename = "_" + orientation_name
long_labels = ["Minimum", "Average", "Maximum"]
plotting_line_long(folder_longitude, KI_long,          "KI against Longitude || Orientation: " + orientation_name, "Longitude (deg)", "KI (MPa . m^0.5)",   long_labels, data_supp[0], factor=1e6, savename=save_folder+"KI"+savename)
plotting_line_long(folder_longitude, KII_long,        "KII against Longitude || Orientation: " + orientation_name, "Longitude (deg)", "KII (MPa . m^0.5)",  long_labels, data_supp[1], factor=1e6, savename=save_folder+"KII"+savename)
plotting_line_long(folder_longitude, KIII_long,      "KIII against Longitude || Orientation: " + orientation_name, "Longitude (deg)", "KIII (MPa . m^0.5)", long_labels, data_supp[2], factor=1e6, savename=save_folder+"KIII"+savename)
plotting_line_long(folder_longitude, KV_long,          "KV against Longitude || Orientation: " + orientation_name, "Longitude (deg)", "KV (MPa . m^0.5)",   long_labels, factor=1e6, savename=save_folder+"KV"+savename)
plotting_line_long(folder_longitude, G_long,            "G against Longitude || Orientation: " + orientation_name, "Longitude (deg)", "G",                  long_labels, data_supp[3], savename=save_folder+"G"+savename)
plotting_line_long(folder_longitude, phi_long,        "phi against Longitude || Orientation: " + orientation_name, "Longitude (deg)", "phi (deg)",          long_labels, savename=save_folder+"phi"+savename)
plotting_line_long(folder_longitude, psi_long,        "psi against Longitude || Orientation: " + orientation_name, "Longitude (deg)", "psi (deg)",          long_labels, savename=save_folder+"psi"+savename)
plotting_line_long(folder_longitude, phi_frac_long, "phi_f against Longitude || Orientation: " + orientation_name, "Longitude (deg)", "phi (deg)",          long_labels, savename=save_folder+"phi_f"+savename)
plotting_line_long(folder_longitude, psi_frac_long, "psi_f against Longitude || Orientation: " + orientation_name, "Longitude (deg)", "psi (deg)",          long_labels, savename=save_folder+"psi_f"+savename)
#plt.show()

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# COLOURS
# all colours are from the Tango colourmap, see:
# http://tango.freedesktop.org/Tango_Icon_Theme_Guidelines#Color_Palette
COLORS = {"butter": ['#fce94f',
                     '#edd400',
                     '#c4a000'],
          "orange": ['#fcaf3e',
                     '#f57900',
                     '#ce5c00'],
          "chocolate": ['#e9b96e',
                        '#c17d11',
                        '#8f5902'],
          "chameleon": ['#8ae234',
                        '#73d216',
                        '#4e9a06'],
          "skyblue": ['#729fcf',
                      '#3465a4',
                      '#204a87'],
          "plum": ['#ad7fa8',
                   '#75507b',
                   '#5c3566'],
          "scarletred": ['#ef2929',
                         '#cc0000',
                         '#a40000'],
          "aluminium": ['#eeeeec',
                        '#d3d7cf',
                        '#babdb6',
                        '#888a85',
                        '#555753',
                        '#2e3436'],
          }

#FONT = {'family': 'Cabin', 'size': 15}
#matplotlib.rc('font', **FONT)


def draw_scanpath(fix_x, fix_y, fix_d, alpha=1, invert_y=False, ydim=None):
    if fix_d is None:
        fix_d = 1
    if invert_y:
        if ydim is None:
            raise RuntimeError('ydim must be provided')
        fix_y = ydim - 1 - fix_y

    for i in range(1, len(fix_x)):
        plt.arrow(fix_x[i - 1], fix_y[i - 1], fix_x[i] - fix_x[i - 1], fix_y[i] - fix_y[i - 1], alpha=alpha,
                  fc=COLORS['chameleon'][0], ec=COLORS['chameleon'][0], fill=True, shape='full', width=3, head_width=0,
                  head_starts_at_zero=False, overhang=0)

    for i in range(len(fix_x)):
        if i == 0:
            plt.plot(fix_x[i], fix_y[i], marker='o', ms=fix_d[i] / 10, mfc=COLORS['skyblue'][0], mec='black', alpha=0.7)
        elif i == len(fix_x) - 1:
            plt.plot(fix_x[i], fix_y[i], marker='o', ms=fix_d[i] / 10, mfc=COLORS['scarletred'][0], mec='black', alpha=0.7)
        else:
            plt.plot(fix_x[i], fix_y[i], marker='o', ms=fix_d[i] / 10, mfc=COLORS['aluminium'][0], mec='black', alpha=0.7)

    for i in range(len(fix_x)):
        plt.text(fix_x[i]-4, fix_y[i]+1, str(i + 1), color='black', ha='left', va='center',
                 multialignment='center', alpha=alpha)


def compute_density_image(points, size=(768, 1024), flip=True):
    #Saliency
    from scipy.ndimage import gaussian_filter
    if flip:
        points = np.flip(points,1)
    sigma=1/0.039
    H, xedges, yedges = np.histogram2d(points[:,0], points[:,1], bins=(range(size[0]+1), range(size[1]+1)))
    Z = gaussian_filter(H, sigma=sigma)
    Z = Z/float(np.sum(Z))
    return Z


def get_saccade_stats(points):
    dirs = []
    amps = []
    nfix = points.shape[0]
    for f in range(nfix-1):
        curr_fix2 = points[f+1,:2]
        curr_fix1 = points[f,:2]
        direction = np.arctan2(curr_fix2[1]-curr_fix1[1], curr_fix2[0]-curr_fix1[0]) + np.pi
        amplitude = np.linalg.norm(curr_fix2-curr_fix1)
        dirs.append(direction)
        amps.append(amplitude)
    return np.rad2deg(np.array(dirs)), np.array(amps)
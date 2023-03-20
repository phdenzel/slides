#!/usr/bin/env python3
"""
Animation scenes for presentations
"""
import sys
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anima
from animobj import AnimObj
from astroobj import Photon, Galaxy, RadioDish, SignalResponse


def populate():
    global objs, imgs
    ax = plt.gca()
    # galaxy
    galaxy = Galaxy('white', position=0+0j, size=0.2, scale=1, flip=False)
    galaxy_img = ax.imshow(galaxy.pixel_array, extent=galaxy.extent)
    # dishes and response graphs
    dish1 = RadioDish('white', position=0.5-0.9j, size=0.2, scale=1, flip=True)
    resp1 = SignalResponse(position=0.4-1.2j)
    dish2 = RadioDish('white', position=0.9-1j, size=0.2, scale=1, flip=True)
    resp2 = SignalResponse(position=0.8-1.4j)
    dish1_img = ax.imshow(dish1.pixel_array, extent=dish1.extent)
    dish2_img = ax.imshow(dish2.pixel_array, extent=dish2.extent)
    resp1_img = resp1.add_graph(flat=True, v=0.2, h=0.6,
                                sigma=0.02, signal=0, gain=0.1, N=300, lw=2)
    resp2_img = resp2.add_graph(flat=True, v=0.2, h=0.6,
                                sigma=0.02, signal=0.4, gain=0.1, N=300, lw=2)
    ax.text(0.5, -1.1, "p", color='white', fontsize=20)
    ax.text(0.9, -1.3, "q", color='white', fontsize=20)
    # gathering all objects and images/plots
    objs = [galaxy, dish1, dish2, resp1, resp2]
    imgs = [galaxy_img, dish1_img, dish2_img, resp1_img, resp2_img]
    xtnt = galaxy.max_extent(*[o.extent for o in objs[1:]])
    ax.set_xlim(xtnt[0], xtnt[1])
    ax.set_ylim(xtnt[2], xtnt[3])


def galaxylight(i, dx):
    global signal_init
    ax = plt.gca()
    for p in objs:
        p.increment()
    pos = 0.05-0.05j
    dish_pos = (objs[1].position + objs[2].position)/2
    # initialize photons
    if N_photons > len(photons):
        photon = Photon(suffix="white", position=pos, size=0.1, scale=pscale)
        theta = np.arctan2(pos.imag, pos.real) / AnimObj.DEG2RAD
        photon.rotate(theta, inplace=True)
        photon.phase_io(-0.99)
        photons.append(photon)
        pimg = ax.imshow(photon.pixel_array, extent=photon.extent)
        pimgs.append(pimg)
        arcimg = photon.add_arc([-5*np.pi/12., -1*np.pi/12.], arc_N=500, alpha=0, lw=2)
        arcimgs.append(arcimg)
    # photon propagation
    for p, photon in enumerate(photons):
        pimg = pimgs[p]
        arcimg = arcimgs[p]
        # move
        photon.move(-dx, radial=True, increment=True)
        pimg.set_data(photon.pixel_array)
        arcimg.set_data(photon.arc)
        arcimg.set(alpha=photon.arc_kwargs['alpha'])
        pimg.set_extent(photon.extent)
        # reset
        # if photon.distance >= 1.1:  # 2*offset:
        #     photon.reset(pos, phase_io=-0.99, size=0.1, scale=pscale)
        # phase in and out
        inc = 0.05
        phasein = int(0.5/inc)
        if photon.steps < phasein:
            photon.phase_shift(inc)
            photon.arc_kwargs['alpha'] = min(1, inc+photon.arc_kwargs['alpha'])
        if photon.distance >= offset:
            photon.phase_shift(-inc)
            photon.arc_kwargs['alpha'] = max(0, photon.arc_kwargs['alpha']-inc)
    # dish reponses
    resps = objs[-2:]
    respimgs = imgs[-2:]
    for r, resp in enumerate(resps):
        respimg = respimgs[r]
        # if photons cross dishes, a new signal is triggered
        new_signal = any([ 0.51-0.005 <= photon.position.real <= 0.51+0.005 for photon in photons])
        signal_init = signal_init or new_signal
        if new_signal:
            resp.graph_signal = 0.2*r
        if signal_init:
            respimg.set_data(resp.graph(flat=False, signal_inc=0.01))


def galaxy_scene():
    # global settings/configurations
    global offset, nframes, verbose
    verbose = True
    aspect = [25, 10]
    # Note: settings not in physical units anymore...
    fps = 60                  # Hz
    duration = 7.2            # s
    nframes = duration * fps  # # of frames
    dt = 1000/fps             # ms
    offset = 0.9              # length units
    dx = 0.005                # length units
    global photons, pimgs, arcimgs, N_photons, pscale
    N_photons = 1
    photons = []
    pimgs = []
    arcimgs = []
    pscale = 1
    global objs, imgs, signal_init
    objs = []
    imgs = []
    signal_init = False
    # set up figure
    plt.style.use('dark_background')
    fig = plt.figure(figsize=aspect)
    # fig.patch.set_alpha(0.0)
    ax = fig.add_subplot(111)
    # ax.set_aspect('equal')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)
    ax.set_axis_off()
    fig.tight_layout()
    # run animation
    nframes = tqdm.tqdm(range(int(0.5*nframes) + 7), file=sys.stdout)
    a = anima.FuncAnimation(
        fig, galaxylight, init_func=populate, fargs=(dx,),
        frames=nframes, interval=dt
    )
    savename = 'scenes/radio_dish_scheme'
    ext = 'webm'
    a.save("{}.{}".format(savename, ext), fps=fps, codec='libvpx-vp9',
           savefig_kwargs={'transparent': True})


if __name__ == "__main__":

    galaxy_scene()

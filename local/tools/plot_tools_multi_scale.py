#!/usr/bin/python
"""
tools for generating interactive plots of the map space (also with multi-scale spectrograms)
"""

import numpy as np
import pandas as pd
import holoviews as hv
from holoviews import opts
from bokeh.models import HoverTool
from bokeh.io import show
from colorcet import glasbey_light
from holoviews.streams import Selection1D
hv.extension('bokeh')
from plot_tools import *

def interactive_with_image_multi_scale(x, y, features, times, scales=[0.5, 1.0, 2.0], scale=2.0, index=None, 
									   title='', vmin=0, vmax=25.5):
	"""Create layout of interactive points plot and corresponding q-transform image.
	"""

	def selected_index(index):
		"""Create image corresponding to selected image in point plot.
		"""

		scale_idx = scales.index(scale)
		# if vmin == 'auto':
		# 	vmin = np.min(x[index[0],scale_idx,:,:])
		# if vmax == 'auto':
		# 	vmax = np.max(x[index[0],scale_idx,:,:])
		if index:
			# # scale_idx = scales.index(scale)
			# if vmin == 'auto':
			# 	vmin = np.min(x[index[0],scale_idx,:,:])
			# if vmax == 'auto':
			# 	vmax = np.max(x[index[0],scale_idx,:,:])
			selected = hv.Image(np.flipud(x[index[0],scale_idx,:,:].transpose())).opts(width=500, height=500, 
																					   cmap='viridis', clim=(vmin, vmax))
			# selected.redim(z=dict(range=(vmin, vmax)))
			label = '%s, %f, %d, %d selected' % (y[index[0]], times[index[0]], index[0], len(index))
		else:
			selected = hv.Image(np.flipud(x[index,scale_idx,:,:].transpose())).opts(width=500, height=500, cmap='viridis')
			label = 'No selection'
		return selected.relabel(label).opts(labelled=[])

	points = interactive_plot(features, y, times, index, title=title)

	selection = Selection1D(source=points)

	xtick_pos = np.linspace(-0.5, 0.5, 11)
	xtick_label = np.linspace(0, scale, 11)
	xticks = []
	for a, b in zip(xtick_pos, xtick_label):
		xticks.append((a, "{:.1f}".format(b)))
	ytick_pos = np.linspace(-0.5, 0.5, 9)
	ytick_label = np.logspace(4, 11, 8, base=2)
	yticks = [(ytick_pos[0], '')]
	for a, b in zip(ytick_pos[1:], ytick_label):
		yticks.append((a, str(int(b))))

	dmap = hv.DynamicMap(selected_index, streams=[selection]).opts(xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), 
																   xticks=xticks, yticks=yticks, 
																   xlabel='time [s]', ylabel='frequency [Hz]')
	layout = (points.opts(toolbar='above') + dmap.opts(toolbar=None)).opts(merge_tools=False)
	return layout

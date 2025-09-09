from functools import partial

import numpy as np

from mne.annotations import _annotations_starts_stops
from mne.filter import create_filter, _overlap_add_filter
from mne.io.pick import (pick_types, _pick_data_channels, pick_info,
                       _PICK_TYPES_KEYS, pick_channels)
from mne.utils import verbose, _ensure_int, _validate_type, _check_option
from mne.defaults import _handle_default
from mne.viz.raw import _setup_browser_selection, _set_custom_selection, _prepare_mne_browse_raw, _plot_raw_traces, \
    _update_raw_data, _pick_bad_channels, _label_clicked, _plot_update_raw_proj, _close_event, _selection_key_press, \
    _selection_scroll
from mne.viz.utils import (_toggle_options, _toggle_proj, _prepare_mne_browse,
                    _plot_raw_onkey, figure_nobar, plt_show,
                    _plot_raw_onscroll, _mouse_click, _find_channel_idx,
                    _select_bads, _get_figsize_from_config,
                    _setup_browser_offsets, _compute_scalings, plot_sensors,
                    _radio_clicked, _set_radio_button, _handle_topomap_bads,
                    _change_channel_group, _plot_annotations, _setup_butterfly,
                    _handle_decim, _setup_plot_projector, _check_cov,
                    _set_ax_label_style, _draw_vert_line, _simplify_float)

def plot_raw(raw, events=None, duration=10.0, start=0.0, n_channels=20,
             bgcolor='w', color=None, bad_color=(1, 0, 0),
             event_color='cyan', scalings=None, remove_dc=True, order=None,
             show_options=False, title=None, show=True, block=False,
             highpass=None, lowpass=None, filtorder=4, clipping=None,
             show_first_samp=False, proj=True, group_by='type',
             butterfly=False, decim='auto', noise_cov=None, event_id=None,
             show_scrollbars=True, verbose=None):
    import matplotlib as mpl
    from mne.io.base import BaseRaw
    color = _handle_default('color', color)
    scalings = _compute_scalings(scalings, raw, remove_dc=remove_dc,
                                 duration=duration)
    _validate_type(raw, BaseRaw, 'raw', 'Raw')
    n_channels = min(len(raw.info['chs']), n_channels)
    _check_option('clipping', clipping, [None, 'clamp', 'transparent'])
    duration = min(raw.times[-1], float(duration))

    # figure out the IIR filtering parameters
    sfreq = raw.info['sfreq']
    # if highpass is not None and highpass <= 0:
    #     raise ValueError('highpass must be > 0, got %s' % (highpass,))
    if highpass is None and lowpass is None:
        ba = filt_bounds = None
    # else:
    #
    #     filtorder = int(filtorder)
    #     if filtorder == 0:
    #         method = 'fir'
    #         iir_params = None
    #     else:
    #         method = 'iir'
    #         iir_params = dict(order=filtorder, output='sos', ftype='butter')
    #     ba = create_filter(np.zeros((1, int(round(duration * sfreq)))),
    #                        sfreq, highpass, lowpass, method=method,
    #                        iir_params=iir_params)
    #     filt_bounds = _annotations_starts_stops(
    #         raw, ('edge', 'bad_acq_skip'), invert=True)

    # make a copy of info, remove projection (for now)
    info = raw.info.copy()
    projs = info['projs']
    info['projs'] = []
    n_times = raw.n_times

    # allow for raw objects without filename, e.g., ICA
    # if title is None:
    #     title = raw._filenames
    #     if len(title) == 0:  # empty list or absent key
    #         title = '<unknown>'
    #     elif len(title) == 1:
    #         title = title[0]
    #     else:  # if len(title) > 1:
    #         title = '%s ... (+ %d more) ' % (title[0], len(title) - 1)
    #         if len(title) > 60:
    #             title = '...' + title[-60:]
    # elif not isinstance(title, str):
    #     raise TypeError('title must be None or a string')
    if events is not None:
        event_times = events[:, 0].astype(float) - raw.first_samp
        event_times /= info['sfreq']
        event_nums = events[:, 2]
    else:
        event_times = event_nums = None

    # reorganize the data in plotting order
    # TODO Refactor this according to epochs.py
    inds = list()
    types = list()
    # for t in ['grad', 'mag']:
    #     inds += [pick_types(info, meg=t, ref_meg=False, exclude=[])]
    #     types += [t] * len(inds[-1])
    # for t in ['hbo', 'hbr']:
    #     inds += [pick_types(info, meg=False, ref_meg=False, fnirs=t,
    #                         exclude=[])]
    #     types += [t] * len(inds[-1])
    pick_kwargs = dict(meg=False, ref_meg=False, exclude=[])
    for key in _PICK_TYPES_KEYS:
        if key not in ['meg', 'fnirs']:
            pick_kwargs[key] = True
            inds += [pick_types(raw.info, **pick_kwargs)]
            types += [key] * len(inds[-1])
            pick_kwargs[key] = False
    inds = np.concatenate(inds).astype(int)
    if not len(inds) == len(info['ch_names']):
        raise RuntimeError('Some channels not classified, please report '
                           'this problem')

    # put them back to original or modified order for natural plotting
    reord = np.argsort(inds)
    types = [types[ri] for ri in reord]
    # if isinstance(order, (np.ndarray, list, tuple)):
    #     # put back to original order first, then use new order
    #     inds = inds[reord][order]
    # elif order is not None:
    #     raise ValueError('Unkown order, should be array-like. '
    #                      'Got "%s" (%s).' % (order, type(order)))
    #
    # if group_by in ['selection', 'position']:
    #     selections, fig_selection = _setup_browser_selection(raw, group_by)
    #     selections = {k: np.intersect1d(v, inds) for k, v in
    #                   selections.items()}
    # elif group_by == 'original':
    #     if order is None:
    #         order = np.arange(len(inds))
    #         inds = inds[reord[:len(order)]]
    # elif group_by != 'type':
    #     raise ValueError('Unknown group_by type %s' % group_by)

    if not isinstance(event_color, dict):
        event_color = {-1: event_color}
    event_color = {_ensure_int(key, 'event_color key'): event_color[key]
                   for key in event_color}
    for key in event_color:
        if key <= 0 and key != -1:
            raise KeyError('only key <= 0 allowed is -1 (cannot use %s)'
                           % key)
    decim, data_picks = _handle_decim(info, decim, lowpass)
    noise_cov = _check_cov(noise_cov, info)

    # set up projection and data parameters
    first_time = raw._first_time if show_first_samp else 0
    start += first_time
    event_id_rev = {val: key for key, val in (event_id or {}).items()}
    units = _handle_default('units', None)
    unit_scalings = _handle_default('scalings', None)

    params = dict(raw=raw, ch_start=0, t_start=start, duration=duration,
                  info=info, projs=projs, remove_dc=remove_dc, ba=ba,
                  n_channels=n_channels, scalings=scalings, types=types,
                  n_times=n_times, event_times=event_times, inds=inds,
                  event_nums=event_nums, clipping=clipping, fig_proj=None,
                  first_time=first_time, added_label=list(), butterfly=False,
                  group_by=group_by, orig_inds=inds.copy(), decim=decim,
                  data_picks=data_picks, event_id_rev=event_id_rev,
                  noise_cov=noise_cov, use_noise_cov=noise_cov is not None,
                  filt_bounds=filt_bounds, units=units, snap_annotations=False,
                  unit_scalings=unit_scalings, use_scalebars=True,
                  show_scrollbars=show_scrollbars)

    if group_by in ['selection', 'position']:
        # params['fig_selection'] = fig_selection
        # params['selections'] = selections
        params['radio_clicked'] = partial(_radio_clicked, params=params)
        # fig_selection.radio.on_clicked(params['radio_clicked'])
        # lasso_callback = partial(_set_custom_selection, params=params)
        # fig_selection.canvas.mpl_connect('lasso_event', lasso_callback)

    _prepare_mne_browse_raw(params, title, bgcolor, color, bad_color, inds,
                            n_channels)

    # plot event_line first so it's in the back
    event_lines = [params['ax'].plot([np.nan], color=event_color[ev_num])[0]
                   for ev_num in sorted(event_color.keys())]

    params['plot_fun'] = partial(_plot_raw_traces, params=params, color=color,
                                 bad_color=bad_color, event_lines=event_lines,
                                 event_color=event_color)

    _plot_annotations(raw, params)

    params['update_fun'] = partial(_update_raw_data, params=params)
    params['pick_bads_fun'] = partial(_pick_bad_channels, params=params)
    params['label_click_fun'] = partial(_label_clicked, params=params)
    params['scale_factor'] = 1.0
    # set up proj button
    opt_button = None
    if len(raw.info['projs']) > 0 and not raw.proj:
        ax_button = params['fig'].add_axes(params['proj_button_pos'])
        ax_button.set_axes_locator(params['proj_button_locator'])
        params['ax_button'] = ax_button
        params['apply_proj'] = proj
        opt_button = mpl.widgets.Button(ax_button, 'Proj')
        callback_option = partial(_toggle_options, params=params)
        opt_button.on_clicked(callback_option)
    # set up callbacks
    callback_key = partial(_plot_raw_onkey, params=params)
    params['fig'].canvas.mpl_connect('key_press_event', callback_key)
    callback_scroll = partial(_plot_raw_onscroll, params=params)
    params['fig'].canvas.mpl_connect('scroll_event', callback_scroll)
    callback_pick = partial(_mouse_click, params=params)
    params['fig'].canvas.mpl_connect('button_press_event', callback_pick)

    # As here eeg_code is shared with plot_evoked, some extra steps:
    # first the actual plot update function
    params['plot_update_proj_callback'] = _plot_update_raw_proj
    # then the toggle handler
    callback_proj = partial(_toggle_proj, params=params)
    # store these for use by callbacks in the options figure
    params['callback_proj'] = callback_proj
    params['callback_key'] = callback_key
    # have to store this, or it could get garbage-collected
    params['opt_button'] = opt_button
    params['update_vertline'] = partial(_draw_vert_line, params=params)

    # do initial plots
    callback_proj('none')

    # deal with projectors
    if show_options:
        _toggle_options(None, params)

    callback_close = partial(_close_event, params=params)
    params['fig'].canvas.mpl_connect('close_event', callback_close)
    # initialize the first selection set
    if group_by in ['selection', 'position']:
        # _radio_clicked(fig_selection.radio.labels[0]._text, params)
        callback_selection_key = partial(_selection_key_press, params=params)
        callback_selection_scroll = partial(_selection_scroll, params=params)
        params['fig_selection'].canvas.mpl_connect('close_event',
                                                   callback_close)
        params['fig_selection'].canvas.mpl_connect('key_press_event',
                                                   callback_selection_key)
        params['fig_selection'].canvas.mpl_connect('scroll_event',
                                                   callback_selection_scroll)
    if butterfly:
        _setup_butterfly(params)

    try:
        plt_show(show, block=block)
    except TypeError:  # not all versions have this
        plt_show(show)

    # add MNE params dict to the resulting figure object so that parameters can
    # be modified after the figure has been created; this is useful e.g. to
    # remove the keyboard shortcut to close the figure with the 'Esc' key,
    # which can be done with
    #
    # fig._mne_params['close_key'] = None
    #
    # (assuming that the figure object is fig)
    params['fig']._mne_params = params

    return params['fig'], params





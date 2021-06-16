import numpy as np

def to_multiscale_timesurface_numpy(
    events,
    sensor_size,
    ordering,
    tau=5e3,
    decay="lin",
    merge_polarities=False
):
    """Representation that creates multiscale timesurfaces for each event for one recording.

    Args:
        tau (float): time constant to decay events around occuring event with.
        decay (str): can be 'lin', 'exp' or 'gau' corresponding to linear, exponential or gaussian decay.
        merge_polarities (bool): flag that tells whether polarities should be taken into account separately or not.

    Returns:
        array of multiscale timesurfaces with dimensions (w,h)
    """

    assert "x" and "y" and "t" and "p" in ordering
    assert len(sensor_size) == 2
    x_index = ordering.find("x")
    y_index = ordering.find("y")
    t_index = ordering.find("t")
    p_index = ordering.find("p")
    n_of_events = len(events)
    if merge_polarities:
        events[:, p_index] = np.zeros(n_of_events)
    n_of_pols = len(np.unique(events[:, p_index]))

    # find number of multiscale levels depending on sensor size
    multiscale_dimensions = np.array([(2**(i+2) - 1) for i in np.arange(10)])
    multiscale_dimensions = multiscale_dimensions[multiscale_dimensions <= np.min(sensor_size)]
    radius = multiscale_dimensions // 2
    dilation_factor = [(multiscale_dimensions[i-1] // 2) + 1 if i > 0 else 1 for i in np.arange(len(multiscale_dimensions))]

    # create multiscale timesurface
    timestamp_memory = np.zeros(
        (n_of_pols, sensor_size[0] + radius[-1] * 2, sensor_size[1] + radius[-1] * 2)
    )
    timestamp_memory -= tau * 3 + 1
    all_surfaces = np.zeros(
        (n_of_events, n_of_pols, len(multiscale_dimensions), multiscale_dimensions[0], multiscale_dimensions[0])
    )

    for index, event in enumerate(events):
        x = int(event[x_index])
        y = int(event[y_index])

        for i, rad in enumerate(radius):
            timestamp_memory[int(event[p_index]), x + rad, y + rad] = event[
                t_index
            ]
            timestamp_context = (
                timestamp_memory[
                    :, x : x + multiscale_dimensions[i], y : y + multiscale_dimensions[i]
                ]
                - event[t_index]
            )

            if decay == "lin":
                timesurface = timestamp_context / (3 * tau) + 1
                timesurface[timesurface < 0] = 0
            elif decay == "exp":
                timesurface = np.exp(timestamp_context / tau)
                timesurface[timestamp_context < (-3 * tau)] = 0
            elif decay == "gau":
                timesurface = (1 / (tau * np.sqrt(2 * np.pi))) * np.exp(- (timestamp_context**2) / (2*tau**2))
                timesurface[timestamp_context < (-3 * tau)] = 0

            # max pooling to avoid increasing parameters
            if i > 0:
                xc = yc = (multiscale_dimensions[i]-1) // 2
                coordinates = np.array([(xc-dilation_factor[i], yc-dilation_factor[i]),
                                        (xc, yc-dilation_factor[i]),
                                        (xc+dilation_factor[i], yc-dilation_factor[i]),
                                        (xc-dilation_factor[i], yc),
                                        (xc, yc),
                                        (xc+dilation_factor[i], yc),
                                        (xc-dilation_factor[i], yc+dilation_factor[i]),
                                        (xc, yc+dilation_factor[i]),
                                        (xc+dilation_factor[i], yc+dilation_factor[i])])
                maxpooled_timesurface = np.zeros(multiscale_dimensions[0]**2)
                for idx, (xd, yd) in enumerate(coordinates):
                    maxpooled_timesurface[idx] = np.max(timesurface[
                        :, xd-radius[i-1]:xd+radius[i-1]+1, yd-radius[i-1]:yd+radius[i-1]+1
                    ])
                timesurface = maxpooled_timesurface.reshape((multiscale_dimensions[0], multiscale_dimensions[0]))
            all_surfaces[index, :, i, :, :] = timesurface
    return all_surfaces

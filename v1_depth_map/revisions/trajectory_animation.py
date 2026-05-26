"""
Functions to create trajectory animations for treadmill experiments.
Shows the trajectory in RS/OF space over time.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import seaborn as sns
import numpy as np


def create_trajectory_animation(
    trials_df,
    output_filename='trajectory_animation.mp4',
    N=50,
    frame_step=10,
    fps=20,
    figsize=(10, 10),
    xlim=(-5, 0.5),
    ylim=(-8, 4.5),
    gridsize=20,
    show_trial_number=True,
    alternate_colors=True,
    start_frame=0,
    end_frame=None
):
    """
    Create an animated video showing the trajectory in RS/OF space over time.
    
    Parameters
    ----------
    trials_df : pd.DataFrame
        DataFrame containing trial data with RS_stim and OF_stim columns
    output_filename : str, optional
        Name of the output MP4 file, by default 'trajectory_animation.mp4'
    N : int, optional
        Number of points in trajectory tail, by default 50
    frame_step : int, optional
        Sample every Nth frame to make video shorter, by default 10
    fps : int, optional
        Frames per second in output video, by default 20
    figsize : tuple, optional
        Figure size (width, height), by default (10, 10)
    xlim : tuple, optional
        X-axis limits (min, max) for log(RS), by default (-5, 0.5)
    ylim : tuple, optional
        Y-axis limits (min, max) for log(OF), by default (-8, 4.5)
    gridsize : int, optional
        Grid size for hexbin plot, by default 20
    show_trial_number : bool, optional
        Whether to display trial number in corner, by default True
    alternate_colors : bool, optional
        Whether to alternate trajectory colors between trials, by default True
    start_frame : int, optional
        Frame to start animation from, by default 0
    end_frame : int, optional
        Frame to end animation at, by default None (end of data)
        
    Returns
    -------
    None
        Saves the animation as an MP4 file
        
    Notes
    -----
    Requires ffmpeg to be installed on the system.
    """
    
    # Prepare data
    all_rs = np.hstack(trials_df.RS_stim.values)
    all_of = np.hstack(trials_df.OF_stim.values)
    valid = (all_rs > 0.01) & (all_of > 0.01)
    
    # Prepare trial-based data
    rs_by_trials = trials_df.RS_stim.values
    of_by_trials = trials_df.OF_stim.values
    
    # Create the base plot once (this won't change)
    g = sns.jointplot(
        x=np.log(all_rs[valid]), 
        y=np.log(all_of[valid]), 
        kind="hex", 
        gridsize=gridsize, 
        xlim=xlim, 
        ylim=ylim
    )
    
    # Add colorbar
    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
    cbar_ax = g.fig.add_axes([.85, .25, .01, .4])
    cb = plt.colorbar(cax=cbar_ax)
    cb.set_label("Number of frames")
    g.ax_joint.set_xlabel("log(RS)")
    g.ax_joint.set_ylabel("log(OF)")
    
    # Initialize line and scatter objects for the trajectory
    line, = g.ax_joint.plot([], [], 'darkorchid', marker='o', linestyle='', mec='none', ms=5, linewidth=2, alpha=0.3)
    scatter = g.ax_joint.scatter([], [], color='darkorchid', s=100, alpha=0.7, zorder=5)
    
    if show_trial_number:
        text = g.ax_joint.text(
            0.02, 0.98, '', 
            transform=g.ax_joint.transAxes, 
            verticalalignment='top', 
            fontsize=12, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
    else:
        text = None
    
    def init():
        """Initialize animation"""
        line.set_data([], [])
        scatter.set_offsets(np.empty((0, 2)))
        if text is not None:
            text.set_text('')
        return (line, scatter, text) if text else (line, scatter)
    
    def animate(frame_id):
        """Update function for animation"""
        # Calculate which trial we're in
        nframes_per_trial = np.hstack([[0], np.cumsum([len(trs) for trs in rs_by_trials])])
        itrial = np.searchsorted(nframes_per_trial, frame_id) - 1
        
        # Make sure we don't go out of bounds
        if itrial >= len(rs_by_trials):
            itrial = len(rs_by_trials) - 1
        
        trs = rs_by_trials[itrial]
        tof = of_by_trials[itrial]
        frame_in_trial = frame_id - nframes_per_trial[itrial]
        
        # Make sure frame_in_trial doesn't exceed trial length
        frame_in_trial = min(frame_in_trial, len(trs))
        
        # Get last N points
        last_n_rs = trs[max(frame_in_trial - N, 0):frame_in_trial]
        last_n_of = tof[max(frame_in_trial - N, 0):frame_in_trial]
        
        # Clip values to plot limits
        last_n_rs = np.clip(last_n_rs, np.exp(xlim[0]), np.inf)
        last_n_of = np.clip(last_n_of, np.exp(ylim[0]), np.inf)
        last_n_rs = np.log(last_n_rs)
        last_n_of = np.log(last_n_of)
        
        # Update line
        line.set_data(last_n_rs, last_n_of)
        
        # Set color
        if alternate_colors:
            color = 'darkorchid' if itrial % 2 == 0 else 'tomato'
        else:
            color = 'darkorchid'
        line.set_color(color)
        
        # Update scatter (current position)
        if len(last_n_rs) > 0:
            scatter.set_offsets([[last_n_rs[-1], last_n_of[-1]]])
            scatter.set_color(color)
        
        # Update text
        if text is not None:
            if frame_id == start_frame:
                text.set_text('')
            else:
                text.set_text(f'Trial {itrial}, Frame {frame_in_trial}/{len(trs)}')
        
        return (line, scatter, text) if text else (line, scatter)
    
    # Create animation
    if end_frame is None:
        end_frame = len(all_rs)
        
    frames = range(start_frame, end_frame, frame_step)
    anim = animation.FuncAnimation(
        g.fig, 
        animate, 
        init_func=init,
        frames=frames, 
        interval=50, 
        blit=True
    )
    
    # Save as MP4
    print(f"Generating video with {len(list(frames))} frames...")
    print(f"This may take a while...")
    writer = FFMpegWriter(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
    anim.save(output_filename, writer=writer)
    print(f"Video saved as {output_filename}")
    
    plt.close()


def plot_trajectory_snapshot(
    trials_df,
    frame_id,
    N=50,
    figsize=(10, 10),
    xlim=(-5, 0.5),
    ylim=(-8, 4.5),
    gridsize=20,
    show_legend=True
):
    """
    Create a single snapshot of the trajectory at a specific frame.
    
    Parameters
    ----------
    trials_df : pd.DataFrame
        DataFrame containing trial data with RS_stim and OF_stim columns
    frame_id : int
        Frame number to plot trajectory up to
    N : int, optional
        Number of points in trajectory tail, by default 50
    figsize : tuple, optional
        Figure size (width, height), by default (10, 10)
    xlim : tuple, optional
        X-axis limits (min, max) for log(RS), by default (-5, 0.5)
    ylim : tuple, optional
        Y-axis limits (min, max) for log(OF), by default (-8, 4.5)
    gridsize : int, optional
        Grid size for hexbin plot, by default 20
    show_legend : bool, optional
        Whether to show legend, by default True
        
    Returns
    -------
    g : seaborn JointGrid
        The plot object
    """
    
    all_rs = np.hstack(trials_df.RS_stim.values)
    all_of = np.hstack(trials_df.OF_stim.values)
    
    valid = (all_rs > 0.01) & (all_of > 0.01)
    g = sns.jointplot(
        x=np.log(all_rs[valid]), 
        y=np.log(all_of[valid]), 
        kind="hex", 
        gridsize=gridsize, 
        xlim=xlim, 
        ylim=ylim
    )
    
    # Add colorbar
    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
    cbar_ax = g.fig.add_axes([.85, .25, .01, .4])
    cb = plt.colorbar(cax=cbar_ax)
    cb.set_label("Number of frames")
    g.ax_joint.set_xlabel("log(RS)")
    g.ax_joint.set_ylabel("log(OF)")
    
    # Plot trajectory of last N points
    if len(all_rs) >= frame_id:
        last_n_rs = np.log(all_rs[max(frame_id - N, 0):frame_id])
        last_n_of = np.log(all_of[max(frame_id - N, 0):frame_id])
        v = valid[max(frame_id - N, 0):frame_id]
        
        g.ax_joint.plot(
            last_n_rs[v], 
            last_n_of[v], 
            'darkorchid',
            marker='o',
            ms=10,
            linestyle='',
            linewidth=2, 
            alpha=0.3, 
            label='__no_legend__'
        )
        g.ax_joint.scatter(
            last_n_rs[v][-1], 
            last_n_of[v][-1], 
            color='darkorchid', 
            s=100,
            alpha=0.7, 
            label=f'Frame {frame_id}'
        )
        
        if show_legend:
            g.ax_joint.legend()
    
    return g

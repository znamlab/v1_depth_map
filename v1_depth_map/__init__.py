from cottage_analysis.preprocessing import synchronisation

from v1_depth_map.figure_utils.utils import *  # noqa: F403

# Monitor frames are matched to the frame log by row number, as when the data were
# analysed. The PZAG22 mouse is the exception: its frame logs start hundreds of frames
# in, so it is matched by FrameIndex.
synchronisation.ROW_MATCHED_PROJECTS.update(
    ["hey2_3d-vision_foodres_20220101", "colasa_3d-vision_revisions"]
)
synchronisation.FRAMEINDEX_MATCHED_PREFIXES.add("PZAG22")

try:
    from v1_depth_map._version import version as __version__
except ImportError:
    __version__ = "unknown"

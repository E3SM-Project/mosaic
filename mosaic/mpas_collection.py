from matplotlib.collections import PolyCollection


class MPASCollection(PolyCollection):
    """
    A PolyCollection designed to mirror patches across periodic boundaries

    Closely follows ``cartopy.mpl.geocollection.GeoCollection`` implementation.
    """

    def get_array(self):
        # Retrieve the array - use copy to avoid any chance of overwrite
        return super().get_array().copy()

    def set_array(self, A):
        # Only use the mirrored indices if they are there
        if hasattr(self, "_mirrored_idxs"):
            self._mirrored_collection_fix.set_array(A[self._mirrored_idxs])

        # Update array for interior patches using underlying implementation
        super().set_array(A)

    def set_clim(self, vmin=None, vmax=None):
        # Update _mirrored_collection_fix color limits if it is there.
        if hasattr(self, "_mirrored_collection_fix"):
            self._mirrored_collection_fix.set_clim(vmin, vmax)

        # Update color limits for the rest of the cells.
        super().set_clim(vmin, vmax)

    def get_datalim(self, transData):
        # TODO: Return corners that were calculated in the polypcolor routine
        # (i.e.: return self._corners). In for the datalims to ignore the
        # extent of mirrored patches.
        return super().get_datalim(transData)

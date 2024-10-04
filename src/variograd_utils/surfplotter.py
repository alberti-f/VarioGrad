# surfplotter module
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


class SurfPlotter:
    """
    Class to plot surfaces with data on them.
    Parameters
    ----------
    lh_surf : str
        Path to the left hemisphere surface file.
    rh_surf : str
        Path to the right hemisphere surface file.
    views : list or str
        List of views to plot. Options are "lateral", "medial", "posterior", "anterior", "dorsal", "ventral".
    layout : str
        Layout of the plots. Options are "grid", "row" or "column".
    
    Attributes
    ----------
    lh_surf : str
        Path to the left hemisphere surface gifti file.
    rh_surf : str
        Path to the right hemisphere surface gifti file.
    views : str, list of str
        List of views to aply to each surface.
        Options are "lateral", "medial", "posterior", "anterior", "dorsal", "ventral".
    layout : str
        Layout of the plots.
        Options are "grid", "row" or "column".
    surf_dict : dict
        Dictionary with the surfaces data.

    Methods
    -------
    plot_surfs(ax=None, lh_map=None, rh_map=None, cmap="viridis", padding=20)
        Plot the surfaces with the data on them.
    _triangle_map(data, hemi)
        Map the data to the triangles of the surface.
    _update_axlim(ax, data)
        Update the limits of the axes.
    _load_surfs(lh_surf, rh_surf)
        Load the surfaces data.
    _apply_transform(points, rotation, traslation)
        Apply the transformation to the points.
    _traslation_params(hemi, v_offset=0, h_offset=0, padding=0)
        Calculate the translation parameters.
    _rotation_matrix(rotation, traslation, scaling)
        Calculate the rotation matrix.
    _rotation_params(hemi, view)
        Calculate the rotation parameters.
    """

    def __init__(self, lh_surf=None, rh_surf=None, views=["lateral", "medial"], layout="grid", zoom=1, padding=20):
        self.lh_surf = lh_surf
        self.rh_surf = rh_surf
        self.views = views if isinstance(views, list) else [views]
        self.layout = layout
        self.zoom = zoom
        self.padding = padding
        self.surf_dict = self._load_surfs(lh_surf=self.lh_surf, rh_surf=self.rh_surf)


    def plot_surfs(self, ax=None, lh_map=None, rh_map=None,
                   cmap="viridis", center=None, cbar=True):
        """
        Plot the data on the surface(s).

        Parameters
        ----------
        ax : matplotlib.axes._subplots.Axes3DSubplot
            Axes to plot the surfaces.
        lh_map : array_like
            Data to plot on the left hemisphere surface.
        rh_map : array_like
            Data to plot on the right hemisphere surface.
        cmap : str
            Colormap to use.
        
        Returns
        -------
        ax : matplotlib.axes._subplots.Axes3DSubplot
            Axes with the surfaces plotted.
        """

        if ax is None:
            _, ax = plt.subplots(1,1, subplot_kw={"projection": "3d"})
        cmap = plt.get_cmap(cmap)
        lsource = LightSource(azdeg=0, altdeg=90)

        maps = dict(zip(["L", "R"],
                        [self._triangle_map(lh_map, "L"), 
                         self._triangle_map(rh_map, "R")]))

        v_offset = 0
        h_offset = 0

        for hemi, surf in self.surf_dict.items():
            if maps[hemi] is None:
                continue

            v_offset = 0 if self.layout=="grid" else v_offset

            for view in self.views:
                scaling = self.zoom
                traslation = self._traslation_params(hemi, v_offset, h_offset)
                rotation = self._rotation_params(hemi, view)
                rotation = self._rotation_matrix(rotation, scaling)
                transformed_pts = self._apply_transform(surf["pts"], rotation, traslation)

                v_offset = transformed_pts.min(axis=0)[2]
                h_offset = transformed_pts.max(axis=0)[1]

                colors, norm = self._norm_data(maps[hemi], center=center)
                poly3d = Poly3DCollection(transformed_pts[surf["trg"], :3], facecolors=cmap(colors), 
                                              edgecolor="none", alpha=1, antialiased=False,
                                              shade=True, lightsource=lsource)
                ax.add_collection3d(poly3d)

                x_lims, y_lims, z_lims = self._update_axlim(ax, transformed_pts)

        self._refine_axis(ax, x_lims, y_lims, z_lims, cbar, cmap, norm)

        return ax


    def _refine_axis(self, ax, x_lims, y_lims, z_lims, cbar, cmap, norm):

        ax.view_init(0, 0)
        ax.set_xlim(x_lims)
        ax.set_ylim(y_lims)
        ax.set_zlim(z_lims)
        ax.set_aspect("equal")
        ax.set_axis_off()

        if cbar:
            if self.layout == "row":
                h = 25
            else:
                h = len(self.views) * 25 if len(self.views) <= 3 else 75

            cax = inset_axes(ax, height=f"{h}%", width="3%", loc="center right")
            plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)


    def _norm_data(self, data, center=None):

        if center is None:
            vmin = data.min()
            vmax = data.max()

        else:
            vmax = np.abs(data - center).max()
            vmin = - vmax

        norm = plt.Normalize(vmin=vmin, vmax=vmax)

        return norm(data), norm


    def _triangle_map(self, data, hemi):
        """
        Map the data to the triangles of the surface computing the average of its vertices.

        Parameters
        ----------
        data : array_like
            Data to map.
        hemi : str
            Hemisphere to map the data.
        
        Returns
        -------
        data : array_like
            Data mapped to the triangles of the surface.
        """

        if data is None:
            return None
        else:
            return data[self.surf_dict[hemi]["trg"]].mean(axis=1)        


    def _update_axlim(self, ax, data):
        """
        Update the limits of the Axes3DSubplot object comparing the extremes of the
        new surface rendering added to the axis.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.Axes3DSubplot
            Axes to update the limits.
        data : array_like
            Data to update the limits.

        Returns
        -------
        x_lims : tuple
            Limits of the x-axis.
        y_lims : tuple
            Limits of the y-axis.
        z_lims : tuple
            Limits of the z-axis.
        """

        update_lims = lambda old_min, old_max, new_min, new_max: (min(old_min, old_max, new_min, new_max), 
                                                                  max(old_min, old_max, new_min, new_max))
        x_lims = update_lims(*ax.get_xlim(), data[:, 0].min(), data[:, 0].max())
        y_lims = update_lims(*ax.get_ylim(), data[:, 1].min(), data[:, 1].max())
        z_lims = update_lims(*ax.get_zlim(), data[:, 2].min(), data[:, 2].max())

        return x_lims, y_lims, z_lims


    def _load_surfs(self, lh_surf, rh_surf):
        """
        Load the surfaces data from the gifti files.
        
        Parameters
        ----------
        lh_surf : str
            Path to the left hemisphere surface gifti file.
        rh_surf : str
            Path to the right hemisphere surface gifti file.
        
        Returns
        -------
        surfs : dict
            Dictionary with the surfaces data.
        """
        keys_to_pop = []
        surfs = {"L": lh_surf, "R": rh_surf}

        for key, value in surfs.items():
            if value is not None:
                surf = nib.load(value).darrays
                surfs[key] = {"pts": surf[0].data - surf[0].data.mean(axis=0), "trg": surf[1].data}

            else:
                keys_to_pop.append(key)

        for key in keys_to_pop:
            surfs.pop(key)

        return surfs


    def _apply_transform(self, points, rotation, traslation):
        """
        Apply the transformation to the points.

        Parameters
        ----------
        points : array_like
            Points to transform.
        rotation : array_like
            Rotation matrix.
        traslation : array_like
            Traslation vector.
        
        Returns
        -------
        transformed : array_like
            Transformed points.
        """

        transformed = np.hstack([points, np.ones([len(points), 1])])
        transformed = transformed @ rotation.T

        if self.layout != "grid":
            traslation[1] -= transformed[:, 1].min() * int(traslation[1]!=0)
        traslation[2] -= transformed[:, 2].max() * int(traslation[2]!=0)
        transformed[:, :3] += traslation

        return transformed[:, :3]


    def _traslation_params(self, hemi, v_offset=0, h_offset=0):
        """
        Calculate the translation parameters.

        Parameters
        ----------
        hemi : str
            Hemisphere to calculate the translation.
        v_offset : int
            Vertical offset.
        h_offset : int
            Horizontal offset.
        
        Returns
        -------
        traslation_vector : array_like
            Translation vector.
        """

        n_surfs = sum([self.lh_surf is not None, self.rh_surf is not None])
        h_offset = h_offset + self.padding
        v_offset = v_offset - self.padding

        if self.layout=="grid":
            if n_surfs==2:
                h_offset = self.surf_dict["L"]["pts"].ptp(axis=0)[1]
                h_offset += h_offset/10 + self.padding
                traslation_vector = [0, 0, v_offset] if hemi=="L" else [0, h_offset, v_offset]
            else:
                traslation_vector = [0, 0, v_offset]

        elif self.layout == "row":
            traslation_vector = [  0, h_offset, 0]

        elif self.layout == "column":
            traslation_vector = [0, 0, v_offset]

        return traslation_vector


    def _rotation_matrix(self, rotation, scaling):
        """
        Calculate the transform matrix for rotation and scaling.

        Parameters
        ----------
        rotation : array_like
            Rotation parameters.
        traslation : array_like
            Traslation vector.
        scaling : int
            Scaling factor.
        
        Returns
        -------
        rotation_matrix : array_like
            Rotation matrix.
        """

        # Convert degrees to radians
        rx, ry, rz = np.radians(rotation)

        # Rotation matrices around x, y, z axes
        Rx = np.array([
            [1, 0, 0, 0],
            [0, np.cos(rx), -np.sin(rx), 0],
            [0, np.sin(rx), np.cos(rx), 0],
            [0, 0, 0, 1]
        ])

        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry), 0],
            [0, 1, 0, 0],
            [-np.sin(ry), 0, np.cos(ry), 0],
            [0, 0, 0, 1]
        ])

        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0, 0],
            [np.sin(rz), np.cos(rz), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Combined rotation matrix
        R = Rz @ Ry @ Rx

        # Scaling matrix
        S = np.array([
            [scaling, 0, 0, 0],
            [0, scaling, 0, 0],
            [0, 0, scaling, 0],
            [0, 0, 0, 1]
        ])

        rotation_matrix = R @ S

        return rotation_matrix


    def _rotation_params(self, hemi, view):
        """
        Calculate the rotation parameters.

        Parameters
        ----------
        hemi : str
            Hemisphere to calculate the rotation.
        view : str
            View to apply.
        
        Returns
        -------
        rotation : array_like
            Rotation parameters.
        """
        hemi_offset = -180 if hemi=="L" else 0
        direction = -1 if hemi=="L" else 1
        z = -180 if view=="lateral" else 0
        if view == "lateral":
            x = 0
            y = 0
            z = hemi_offset
        elif view == "medial":
            x = 0
            y = 0
            z = hemi_offset + 180
        elif view == "posterior":
            x = 0
            y = 0
            z = hemi_offset + (90 * direction)
        elif view == "anterior":
            x = 0
            y = 0
            z = hemi_offset - (90 * direction)
        elif view == "dorsal":
            x = 90
            y = 0
            z = hemi_offset + (90 * direction)
        elif view == "ventral":
            x = 90
            y = 0
            z = hemi_offset - (90 * direction)
        elif isinstance(view, (list, tuple)):
            x, y, z = view
            y *= direction
            z = hemi_offset + (direction * z)
        else:
            raise ValueError("Unknown view.")


        return [x, y, z]
    
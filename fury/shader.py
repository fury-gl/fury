"""Shader utilities function Module."""

from numpy import ceil, prod

from fury.lib import (
    BaseShader,
    Binding,
    Buffer,
    ThinLineSegmentShader,
    load_wgsl,
)


class VectorFieldComputeShader(BaseShader):
    """Compute shader for vector field.

    Parameters
    ----------
    wobject : VectorField
        The vector field object to be rendered.
    """

    type = "compute"

    def __init__(self, wobject):
        """Initialize the vector field compute shader.

        Parameters
        ----------
        wobject : VectorField
            The vector field object to be rendered.
        """
        super().__init__(wobject)
        self["num_vectors"] = wobject.vectors_per_voxel
        self["data_shape"] = wobject.field_shape
        self["workgroup_size"] = 1024

    def get_render_info(self, wobject, _shared):
        """Get render information for the vector field compute shader.

        Parameters
        ----------
        wobject : VectorField
            The vector field object to be rendered.
        _shared : dict
            Shared information for the shader.

        Returns
        -------
        dict
            A dictionary containing the render information.
        """
        n = int(ceil(prod(wobject.field_shape) / self["workgroup_size"]))
        return {
            "indices": (n, 1, 1),
        }

    def get_pipeline_info(self, _wobject, _shared):
        """Get pipeline information for the vector field compute shader.

        Parameters
        ----------
        _wobject : VectorField
            The vector field object to be rendered.
        _shared : dict
            Shared information for the shader.

        Returns
        -------
        dict
            A dictionary containing pipeline information.
        """
        return {}

    def get_bindings(self, wobject, _shared):
        """Get the bindings for the vector field compute shader.

        Parameters
        ----------
        wobject : VectorField
            The vector field object to be rendered.
        _shared : dict
            Shared information for the shader.

        Returns
        -------
        dict
            A dictionary containing the bindings for the shader.
        """
        # To share the bindings across compute and render shaders, we need to
        # define the bindings exactly the same way in both shaders.
        bindings = {
            0: Binding(
                "s_vectors", "buffer/storage", Buffer(wobject.vectors), "COMPUTE"
            ),
            1: Binding("s_scales", "buffer/storage", Buffer(wobject.scales), "COMPUTE"),
            3: Binding(
                "s_positions", "buffer/storage", wobject.geometry.positions, "COMPUTE"
            ),
            4: Binding(
                "s_colors", "buffer/storage", wobject.geometry.colors, "COMPUTE"
            ),
        }
        self.define_bindings(0, bindings)

        return {0: bindings}

    def get_code(self):
        """Get the WGSL code for the vector field compute shader.

        Returns
        -------
        str
            The WGSL code as a string.
        """
        return load_wgsl("vector_field_compute.wgsl", package_name="fury.wgsl")


class VectorFieldShader(ThinLineSegmentShader):
    """Shader for VectorFieldActor.

    Parameters
    ----------
    wobject : VectorField
        The vector field object to be rendered.
    """

    def __init__(self, wobject):
        """Initialize the VectorFieldShader with the given vector field object.

        Parameters
        ----------
        wobject : VectorField
            The vector field object to be rendered.
        """
        super().__init__(wobject)
        self["num_vectors"] = wobject.vectors_per_voxel
        self["data_shape"] = wobject.field_shape

    def get_code(self):
        """Get the WGSL code for the vector field render shader.

        Returns
        -------
        str
            The WGSL code as a string.
        """
        return load_wgsl("vector_field_render.wgsl", package_name="fury.wgsl")

"""Shader utilities function Module."""

from numpy import ceil, prod

from fury.lib import (
    BaseShader,
    Binding,
    Buffer,
    LineShader,
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
        self["workgroup_size"] = 64

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


class VectorFieldThinShader(ThinLineSegmentShader):
    """Shader for VectorFieldActor.

    Parameters
    ----------
    wobject : VectorField
        The vector field object to be rendered.
    """

    def __init__(self, wobject):
        """Initialize the VectorFieldThinLineShader with the given vector field object.

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
        return load_wgsl("vector_field_thin_render.wgsl", package_name="fury.wgsl")


class VectorFieldShader(LineShader):
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
        self["line_type"] = "segment"

    def get_code(self):
        """Get the WGSL code for the vector field render shader.

        Returns
        -------
        str
            The WGSL code as a string.
        """
        return load_wgsl("vector_field_render.wgsl", package_name="fury.wgsl")


class VectorFieldArrowShader(VectorFieldShader):
    """Shader for VectorFieldArrowActor.

    Parameters
    ----------
    wobject : VectorField
        The vector field object to be rendered.
    """

    def __init__(self, wobject):
        """Initialize the VectorFieldArrowShader with the given vector field object.

        Parameters
        ----------
        wobject : VectorField
            The vector field object to be rendered.
        """
        super().__init__(wobject)
        self["line_type"] = "arrow"


class SphGlyphComputeShader(BaseShader):
    type = "compute"

    def __init__(self, wobject):
        super().__init__(wobject)
        self["n_coeffs"] = wobject.n_coeff
        self["vertices_per_glyph"] = wobject.vertices_per_glyph
        self["faces_per_glyph"] = wobject.faces_per_glyph
        self["data_shape"] = wobject.data_shape
        self["workgroup_size"] = (1536, 1, 1)
        self["n_vertices"] = prod(wobject.data_shape) * wobject.vertices_per_glyph

    def get_render_info(self, wobject, _shared):
        n = int(ceil(prod(wobject.data_shape) / prod(self["workgroup_size"])))
        print("n", n)
        return {
            "indices": (n, 1, 1),
        }

    def get_pipeline_info(self, _wobject, _shared):
        return {}

    def get_bindings(self, wobject, shared):
        geometry = wobject.geometry

        bindings = {
            0: Binding(
                "s_coeffs", "buffer/storage", Buffer(wobject.sh_coeff), "COMPUTE"
            ),
            1: Binding(
                "s_sf_func", "buffer/storage", Buffer(wobject.sf_func), "COMPUTE"
            ),
            2: Binding("s_sphere", "buffer/storage", Buffer(wobject.sphere), "COMPUTE"),
            3: Binding(
                "s_indices", "buffer/storage", Buffer(wobject.indices), "COMPUTE"
            ),
            4: Binding("s_positions", "buffer/storage", geometry.positions, "COMPUTE"),
            5: Binding("s_normals", "buffer/storage", geometry.normals, "COMPUTE"),
            6: Binding("s_colors", "buffer/storage", geometry.colors, "COMPUTE"),
            7: Binding(
                "s_scaled_vertice",
                "buffer/storage",
                Buffer(wobject.scaled_vertices),
                "COMPUTE",
            ),
        }
        self.define_bindings(0, bindings)
        return {
            0: bindings,
        }

    def get_code(self):
        return load_wgsl("sph_glyph_compute.wgsl", package_name="fury.wgsl")

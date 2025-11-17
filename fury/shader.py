"""Shader utilities function Module."""

from numpy import ceil, prod

from fury.lib import (
    BaseShader,
    Binding,
    Buffer,
    LineShader,
    MeshPhongShader,
    MeshShader,
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


class StreamlinesShader(LineShader):
    """Shader for StreamlineActor."""

    def get_code(self):
        """Get the WGSL code for the streamline render shader.

        Returns
        -------
        str
            The WGSL code as a string.
        """
        return load_wgsl("streamline_render.wgsl", package_name="fury.wgsl")


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
    """Compute shader for spherical harmonics glyph rendering.

    Parameters
    ----------
    wobject : SphGlyph
        The spherical glyph object to be rendered.
    """

    type = "compute"

    def __init__(self, wobject):
        """Initialize the SphGlyphComputeShader with the given spherical glyph object.

        Parameters
        ----------
        wobject : SphGlyph
            The spherical glyph object to be rendered.
        """
        super().__init__(wobject)
        self["n_coeffs"] = wobject.n_coeff
        self["vertices_per_glyph"] = wobject.vertices_per_glyph
        self["faces_per_glyph"] = wobject.faces_per_glyph
        self["data_shape"] = wobject.data_shape
        self["workgroup_size"] = (64, 1, 1)
        self["n_vertices"] = prod(wobject.data_shape) * wobject.vertices_per_glyph
        self["color_type"] = wobject.color_type

    def get_render_info(self, wobject, _shared):
        """Get the render information for the spherical glyph.

        Parameters
        ----------
        wobject : SphGlyph
            The spherical glyph object to be rendered.
        _shared : dict
            Shared information for the shader.

        Returns
        -------
        dict
            A dictionary containing the render information.
        """
        n = int(ceil(prod(wobject.data_shape) / prod(self["workgroup_size"])))
        return {
            "indices": (n, 1, 1),
        }

    def get_pipeline_info(self, _wobject, _shared):
        """Get pipeline information for the spherical harmonic glyph compute shader.

        Parameters
        ----------
        _wobject : SphGlyph
            The spherical glyph object to be rendered.
        _shared : dict
            Shared information for the shader.

        Returns
        -------
        dict
            A dictionary containing pipeline information.
        """
        return {}

    def get_bindings(self, wobject, _shared):
        """Get the bindings for the spherical harmonic glyph compute shader.

        Parameters
        ----------
        wobject : SphGlyph
            The spherical glyph object to be rendered.
        _shared : dict
            Shared information for the shader.

        Returns
        -------
        dict
            A dictionary containing the bindings for the shader.
        """
        geometry = wobject.geometry
        material = wobject.material

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
            8: Binding(
                "u_material", "buffer/uniform", material.uniform_buffer, "COMPUTE"
            ),
        }
        self.define_bindings(0, bindings)
        return {
            0: bindings,
        }

    def get_code(self):
        """Get the WGSL code for the spherical harmonic glyph compute shader.

        Returns
        -------
        str
            The WGSL code as a string.
        """
        return load_wgsl("sph_glyph_compute.wgsl", package_name="fury.wgsl")


class LineProjectionComputeShader(BaseShader):
    """Initialize the line projection compute shader.

    Parameters
    ----------
    wobject : LineProjection
        The line projection object to be rendered.
    """

    type = "compute"

    def __init__(self, wobject):
        """Initialize the line projection compute shader.

        Parameters
        ----------
        wobject : LineProjection
            The line projection object to be rendered.
        """
        super().__init__(wobject)
        self["num_lines"] = wobject.num_lines
        self["workgroup_size"] = 64

    def get_pipeline_info(self, _wobject, _shared):
        """Get pipeline information for the shader.

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

    def get_render_info(self, wobject, _shared):
        """Get render information for the shader.

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
        n = int(ceil(wobject.num_lines / self["workgroup_size"]))
        return {
            "indices": (n, 1, 1),
        }

    def get_bindings(self, wobject, _shared):
        """Get the bindings for the line projection compute shader.

        Parameters
        ----------
        wobject : LineProjection
            The line projection object to be rendered.
        _shared : dict
            Shared information for the shader.

        Returns
        -------
        dict
            A dictionary containing the bindings for the shader.
        """
        bindings = {
            0: Binding("s_lines", "buffer/storage", Buffer(wobject.lines), "COMPUTE"),
            1: Binding(
                "u_wobject", "buffer/uniform", wobject.uniform_buffer, "COMPUTE"
            ),
            2: Binding(
                "s_offsets", "buffer/storage", Buffer(wobject.offsets), "COMPUTE"
            ),
            3: Binding(
                "s_positions",
                "buffer/storage",
                wobject.geometry.positions,
                "COMPUTE",
            ),
            4: Binding(
                "s_lengths", "buffer/storage", Buffer(wobject.lengths), "COMPUTE"
            ),
            5: Binding(
                "s_colors", "buffer/storage", wobject.geometry.colors, "COMPUTE"
            ),
            6: Binding(
                "s_edge_colors",
                "buffer/storage",
                wobject.geometry.edge_colors,
                "COMPUTE",
            ),
        }
        self.define_bindings(0, bindings)
        return {
            0: bindings,
        }

    def get_code(self):
        """Get the WGSL code for the shader.

        Returns
        -------
        str
            The WGSL code as a string.
        """
        return load_wgsl("line_projection_compute.wgsl", package_name="fury.wgsl")


class BillboardShader(MeshShader):
    """Shader for Billboard actor.

    Parameters
    ----------
    wobject : Mesh
        The mesh object containing billboard data.
    """

    def __init__(self, wobject):
        """Initialize the BillboardShader with the given mesh object.

        Parameters
        ----------
        wobject : Mesh
            The mesh object containing billboard data.
        """
        super().__init__(wobject)
        if hasattr(wobject, "billboard_count"):
            self["billboard_count"] = wobject.billboard_count
        else:
            self["billboard_count"] = 1

    def get_code(self):
        """Get the WGSL code for the billboard render shader.

        Returns
        -------
        str
            The WGSL code as a string.
        """
        return load_wgsl("billboard_render.wgsl", package_name="fury.wgsl")


class _StreamtubeBakingShader(BaseShader):
    """Internal compute shader that generates streamtube geometry on the GPU.

    This shader is used internally by the streamtube actor when GPU compute
    shaders are available. Users should not instantiate this directly.

    Parameters
    ----------
    wobject : Mesh
        Mesh containing buffers produced by the internal streamtube function.

    Notes
    -----
    This is an internal class marked with a leading underscore. It should not
    be used directly by end users.
    """

    type = "compute"

    def __init__(self, wobject):
        """Initialise the compute shader state for the provided mesh.

        Parameters
        ----------
        wobject : Mesh
            Mesh containing preallocated geometry and line buffers.
        """

        super().__init__(wobject)
        if not hasattr(wobject, "_needs_gpu_update"):
            wobject._needs_gpu_update = True
        self["n_lines"] = wobject.n_lines
        self["max_line_length"] = wobject.max_line_length
        self["tube_sides"] = wobject.tube_sides
        self["tube_radius"] = float(wobject.material.radius)
        self["workgroup_size"] = min(64, max(int(wobject.n_lines), 1))
        self["end_caps"] = 1 if getattr(wobject, "end_caps", False) else 0
        self["color_channels"] = getattr(wobject, "color_components", 3)

    def get_render_info(self, wobject, _shared):
        """Return the dispatch dimensions for the compute shader.

        Parameters
        ----------
        wobject : Mesh
            The mesh to render.
        _shared : dict
            Shared pipeline state (unused).

        Returns
        -------
        dict
            Dictionary containing ``indices`` dispatch dimensions.
        """
        needs_update = getattr(wobject, "_needs_gpu_update", True)
        n_lines = int(wobject.n_lines)
        if not needs_update or n_lines == 0:
            return {"indices": (0, 1, 1)}
        workgroup_size = min(64, max(n_lines, 1))
        groups = int(ceil(n_lines / workgroup_size))
        wobject._needs_gpu_update = False
        return {"indices": (groups, 1, 1)}

    def get_pipeline_info(self, _wobject, _shared):
        """Return additional pipeline information.

        Parameters
        ----------
        _wobject : Mesh
            The mesh to render (unused).
        _shared : dict
            Shared pipeline state (unused).

        Returns
        -------
        dict
            Empty dictionary since no extra pipeline state is required.
        """

        return {}

    def get_bindings(self, wobject, _shared):
        """Describe storage buffers used by the compute shader.

        Parameters
        ----------
        wobject : Mesh
            Mesh whose buffers are bound to the compute shader.
        _shared : dict
            Shared pipeline state (unused).

        Returns
        -------
        dict
            Mapping of bind group to :class:`Binding` definitions.
        """

        geometry = wobject.geometry

        self["n_lines"] = wobject.n_lines
        self["max_line_length"] = wobject.max_line_length
        self["tube_sides"] = wobject.tube_sides
        self["tube_radius"] = float(wobject.material.radius)
        self["workgroup_size"] = min(64, max(int(wobject.n_lines), 1))
        self["color_channels"] = getattr(wobject, "color_components", 3)
        self["end_caps"] = 1 if getattr(wobject, "end_caps", False) else 0

        bindings = {
            0: Binding(
                "s_line_data",
                "buffer/storage",
                wobject.line_buffer,
                "COMPUTE",
            ),
            1: Binding(
                "s_line_lengths",
                "buffer/storage",
                wobject.length_buffer,
                "COMPUTE",
            ),
            2: Binding(
                "s_line_colors",
                "buffer/storage",
                wobject.color_buffer,
                "COMPUTE",
            ),
            3: Binding(
                "s_vertex_offsets",
                "buffer/storage",
                wobject.vertex_offset_buffer,
                "COMPUTE",
            ),
            4: Binding(
                "s_triangle_offsets",
                "buffer/storage",
                wobject.triangle_offset_buffer,
                "COMPUTE",
            ),
            5: Binding(
                "s_vertex_positions",
                "buffer/storage",
                geometry.positions,
                "COMPUTE",
            ),
            6: Binding(
                "s_vertex_normals",
                "buffer/storage",
                geometry.normals,
                "COMPUTE",
            ),
            7: Binding(
                "s_vertex_colors",
                "buffer/storage",
                geometry.colors,
                "COMPUTE",
            ),
            8: Binding(
                "s_indices",
                "buffer/storage",
                geometry.indices,
                "COMPUTE",
            ),
        }
        self.define_bindings(0, bindings)
        return {0: bindings}

    def get_code(self):
        """Load the WGSL source for the streamtube compute shader.

        Returns
        -------
        str
            WGSL shader source for compute dispatch.
        """

        return load_wgsl("streamtube_compute.wgsl", package_name="fury.wgsl")


class _StreamtubeRenderShader(MeshPhongShader):
    """Render shader wrapper that auto-detaches compute after first bake.

    This shader is used internally by the streamtube actor to automatically
    swap from compute-based material to standard render-only material after
    the first baking pass is complete.

    Notes
    -----
    This is an internal class marked with a leading underscore. It should not
    be used directly by end users.
    """

    def get_render_info(self, wobject, shared):
        """Get render info and auto-detach compute shader if baking is done.

        Parameters
        ----------
        wobject : Mesh
            The mesh object being rendered.
        shared : dict
            Shared rendering state.

        Returns
        -------
        dict
            Render information for the shader.
        """
        try:
            # Import here to avoid circular dependency
            from fury.material import StreamtubeMaterial, _StreamtubeBakedMaterial

            mat = getattr(wobject, "material", None)
            needs_update = bool(getattr(wobject, "_needs_gpu_update", False))
            auto_detach = bool(getattr(mat, "auto_detach", True))
            if (
                isinstance(mat, _StreamtubeBakedMaterial)
                and auto_detach
                and not needs_update
            ):
                baked_mat = StreamtubeMaterial(
                    opacity=float(getattr(mat, "opacity", 1.0)),
                    pick_write=bool(getattr(mat, "pick_write", True)),
                    flat_shading=bool(getattr(mat, "flat_shading", False)),
                    color_mode=str(getattr(mat, "color_mode", "vertex")),
                )
                wobject.material = baked_mat
                for attr in (
                    "line_buffer",
                    "length_buffer",
                    "color_buffer",
                    "vertex_offset_buffer",
                    "triangle_offset_buffer",
                ):
                    if hasattr(wobject, attr):
                        delattr(wobject, attr)
        except Exception:
            pass
        return super().get_render_info(wobject, shared)

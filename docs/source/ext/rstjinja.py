"""Render explicitly selected RST documents with Jinja for HTML builds."""


def rstjinja(app, docname, source):
    """
    Render a selected document with the configured HTML context.

    Parameters
    ----------
    app : sphinx.application.Sphinx
        Application providing the builder and configuration.
    docname : str
        Sphinx document name, relative to the source root without its suffix.
    source : list of str
        Source text, replaced in place after successful rendering.

    Returns
    -------
    None
        Unselected documents and non-HTML sources remain unchanged.

    Raises
    ------
    jinja2.TemplateError
        If rendering a selected document fails.
    """
    if app.builder.format != "html" or docname not in app.config.rstjinja_documents:
        return
    rendered = app.builder.templates.render_string(source[0], app.config.html_context)
    source[0] = rendered


def setup(app):
    """
    Register the document selection and source renderer.

    Parameters
    ----------
    app : sphinx.application.Sphinx
        Application receiving the configuration and event handler.

    Returns
    -------
    None
    """
    app.add_config_value("rstjinja_documents", [], "env")
    app.connect("source-read", rstjinja)

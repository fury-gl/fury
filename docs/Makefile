# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =   # "-W" This flag turns warnings into errors.
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = FURY
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

clean:
	rm -rf "$(SOURCEDIR)/api"
	rm -rf "$(SOURCEDIR)/auto_examples"
	rm -rf "$(SOURCEDIR)/auto_tutorials"
	rm -rf "$(SOURCEDIR)/reference"
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) clean

html-no-examples:
	$(SPHINXBUILD) -D plot_gallery=0 -b html $(SPHINXOPTS) "$(SOURCEDIR)" "$(BUILDDIR)/html"
	cp -r $(BUILDDIR)/html $(BUILDDIR)/html-web-only
	rm -rf $(BUILDDIR)/html-web-only/auto_examples $(BUILDDIR)/html-web-only/_sources/auto_examples
	rm -rf $(BUILDDIR)/html-web-only/auto_tutorials $(BUILDDIR)/html-web-only/_sources/auto_tutorials
	rm -rf $(BUILDDIR)/html-web-only/reference $(BUILDDIR)/html-web-only/_sources/reference
	rm -rf $(BUILDDIR)/html-web-only/_downloads
	rm -rf $(BUILDDIR)/html-web-only/_images/sphx_glr_*

	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

upload:
	python upload_to_gh-pages.py

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@echo $@
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)."

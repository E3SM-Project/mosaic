mosaic_logo.png: mosaic_logo.nc mosaic_logo.cfg mosaic_logo.py
	./mosaic_logo.py render

mosaic_logo.nc: mosaic_logo.cfg mosaic_logo.py check-4-jigsaw
	./mosaic_logo.py mesh

check-4-jigsaw:
	@command -v jigsaw >/dev/null 2>&1 || $(MAKE) build-jigsaw

build-jigsaw:
	build_jigsaw --clone

clean:
	rm -f mosaic_logo.png
	rm -f mosaic_logo.nc
	rm -rf jigsaw_workdir

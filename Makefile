PREFIX ?= $(HOME)/phdenzel.github.io/assets/blog-assets/xyz-blog-title
ROOT_DIR := $(shell git rev-parse --show-toplevel)

.PHONY: export
export: emacs-compile
	mv index.html slides.html
	sed -i 's#file:///home/phdenzel/local/reveal.js/dist/reveal.css#./assets/css/reveal.css#g' slides.html
	sed -i 's#file:///home/phdenzel/local/reveal.js/dist/theme/phdcolloq.css#./assets/css/theme/phdcolloq.css#g' slides.html
	sed -i 's#/home/phdenzel/local/reveal.js/dist/reveal.js#./assets/js/reveal.js#g' slides.html
	sed -i 's#file:///home/phdenzel/local/reveal.js/plugin/markdown/markdown.js#./assets/js/markdown/markdown.js#g' slides.html
	sed -i 's#file:///home/phdenzel/local/reveal.js/plugin/math/math.js#./assets/js/math/math.js#g' slides.html
	sed -i 's#file:///home/phdenzel/local/reveal.js/plugin/zoom/zoom.js#./assets/js/zoom/zoom.js#g' slides.html

.PHONY: emacs-compile
emacs-compile:
	@emacsclient --alternate-editor='' -e '(progn (find-file "index.org") (org-reveal-export-to-html))'

assets/movies:
	@echo "Visit https://mega.nz/folder/NPoXRZqC#NPRDJ64WKJ6mycpy395x0g"
	@echo "Download the folder to $(PWD)/assets/"

.PHONY: export
install: export
	cp slides.html $(PREFIX)/slides.html

assets:
	mkdir -p assets/{css,images,movies}
	@for dpath in $(ROOT_DIR)/assets/*; do \
		echo -n "Copy(y)/walk(w) $${dpath##*/}? [y/N/w] " && \
		read ans && \
		if [ $${ans:-'N'} = 'y' ]; then \
			cp -r "$${dpath}"/* "assets/$${dpath##*/}/"; \
		elif [ $${ans:-'N'} = 'w' ]; then \
			for ipath in $${dpath}/*; do \
				echo -n "Copy(y)/link(l) $${ipath##*/}? [y/N/l] " && \
				read ians && \
				if [ $${ians:-'N'} = 'y' ]; then \
					cp -r "$${dpath}/$${ipath##*/}" "$(PWD)/assets/$${dpath##*/}/$${ipath##*/}"; \
				elif [ $${ians:-'N'} = 'l' ]; then \
					realpath --relative-to="assets/$${dpath##*/}" "$${dpath}" > rpath; \
					ln -s "$$(cat rpath)/$${ipath##*/}" "$(PWD)/assets/$${dpath##*/}/$${ipath##*/}"; rm -f rpath; \
				fi; \
			done; \
		fi; \
	done

.PHONY: clean
clean:
	rm -rf index.html

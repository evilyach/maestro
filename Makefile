init:
	./scripts/makefile-init.sh

lint:
	./scripts/makefile-lint.sh

wheel:
	./scripts/makefile-wheel.sh

debug: wheel
	./scripts/makefile-debug.sh

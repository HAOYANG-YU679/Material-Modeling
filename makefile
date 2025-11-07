WIDTH ?= 10

THETA ?= 15

carbon:
	python carbon.py --plot $(WIDTH)

golf:
	python golf.py --plot $(THETA)



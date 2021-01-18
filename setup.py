import setuptools

setuptools.setup(
        name = 'hiragana',
        packages = setuptools.find_packages(),
        install_requires=[
            'numpy', 'torch', 'pillow', 'svgwrite', 'cairosvg', 'matplotlib', 'tqdm'],)

# Dataset for ICDAR 21 paper “Vectorization of Historical Maps Using Deep Edge Filtering and Closed Shape Extraction”

Version 1.0.0 (2021-04-08)

This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).


## Warning
Please note that this dataset is different from the one of the ICDAR 2021 Competition on Historical Map Segmentation (MapSeg) available at https://icdar21-mapseg.github.io/ (Task 1 in particular) as:

1. it contains fewer images,
2. much more closed shapes are annotated, not only building blocks.


## Authors
Names

- Yizi Chen$^{1,2}$ — ORCID: [0000-0003-1637-0092](https://orcid.org/0000-0003-1637-0092)
- Edwin Carlinet$^{1}$ — ORCID: [0000-0001-5737-5266](https://orcid.org/0000-0001-5737-5266)
- Joseph Chazalon$^{1}$ — ORCID: [0000-0002-3757-074X](https://orcid.org/0000-0002-3757-074X)
- Clément Mallet$^{2}$ — ORCID: [0000-0002-2675-165X](https://orcid.org/0000-0002-2675-165X)
- Bertrand Duménieu$^{3}$ — ORCID: [0000-0002-2517-2058](https://orcid.org/0000-0002-2517-2058)
- Julien Perret$^{2,3}$ — ORCID: [0000-0002-0685-0730](https://orcid.org/0000-0002-0685-0730)

Affiliations

1. EPITA Research and Development Lab. (LRDE), EPITA, France
2. Univ. Gustave Eiffel, IGN-ENSG, LaSTIG
3. LaDéHiS, CRH, EHESS


## Funding and Acknowledgements
This work was partially funded by the French National Research Agency (ANR): 
Project SoDuCo, grant ANR-18-CE38-0013.
We thank the City of Paris for granting us with the permission to use and reproduce the atlases used in this work.


## Original image credit
Input images were extracted from the two following resources, with permission.

For `BHdV_PL_ATL20Ardt_1898_0004-*` images, there are based on Sheet 3 (“planche 3”) of the following atlas:

> Atlas municipal des vingt arrondissements de Paris. 1898.
> Bibliothèque de l’Hôtel de Ville. 
> Ville de Paris. France.
> https://bibliotheques-specialisees.paris.fr/ark:/73873/pf0000935509


For `BHdV_PL_ATL20Ardt_1926_0004-*` images, there are based on Sheet 1 (“planche 1”) of the following atlas:

> Atlas municipal des vingt arrondissements de Paris. 1925.
> Bibliothèque de l’Hôtel de Ville. 
> Ville de Paris. France.
> https://bibliotheques-specialisees.paris.fr/ark:/73873/pf0000935524


## Content

MD5 sums for all files (except `README.md`):
```
f8c358fb4849380247a6ec5adecb8736  BHdV_PL_ATL20Ardt_1898_0004-TEST-EDGE_target.png
e57a9bbadbbf0f95411844192843e300  BHdV_PL_ATL20Ardt_1898_0004-TEST-INPUT_black_border.jpg
b0824f2396b8cf966cb4e63aecde599e  BHdV_PL_ATL20Ardt_1898_0004-TEST-INPUT_color_border.jpg
d60d082a5d65054805d991ccef3913ff  BHdV_PL_ATL20Ardt_1898_0004-TEST-MASK_content.png
d5eb1f85020c66437e4440de78ffa775  BHdV_PL_ATL20Ardt_1926_0004-TRAIN-EDGE_target.png
bdb90243a81c8d7bd1e61ad5cd662228  BHdV_PL_ATL20Ardt_1926_0004-TRAIN-INPUT_black_border.jpg
fb9637570e860ada3f59ae9f0c02b799  BHdV_PL_ATL20Ardt_1926_0004-TRAIN-INPUT_color_border.jpg
36654b263420214ba7bcb208e2f6a667  BHdV_PL_ATL20Ardt_1926_0004-TRAIN-MASK_content.png
69b85a8a73212b01d4c6f43a52738a7d  BHdV_PL_ATL20Ardt_1926_0004-VAL-EDGE_target.png
6ee4d9c927a2925420cc7261b134a099  BHdV_PL_ATL20Ardt_1926_0004-VAL-INPUT_black_border.jpg
8251c9e8b28ac462b4557b5c8b30bfe6  BHdV_PL_ATL20Ardt_1926_0004-VAL-INPUT_color_border.jpg
da6dcfb4d972ed04e23df873d0280b71  BHdV_PL_ATL20Ardt_1926_0004-VAL-MASK_content.png
```

The dataset is separated in three subsets, each on made of a single image:

| Subset     | Pattern     | Source                    | Image size  | # shapes | # tiles 500² |
| ---------- | ----------- | ------------------------- | ----------- | -------- | ------------ |
| train      | `*-TRAIN-*` | 1925 atlas, sheet 1 upper | 4500 × 9000 | 3343     | 703          |
| validation | `*-VAL-*`   | 1925 atlas, sheet 1 lower | 3000 × 9000 | 2183     | 481          |
| test       | `*-TEST-*`  | 1898 atlas, sheet 4       | 6000 × 5500 | 2836     | 575          |

Inputs are composed of:

- cropped image content (RGB JPEG), with either 
  - a black border: `*-INPUT_black_border.jpg` images
  - or a colored border: `*-INPUT_color_border.jpg` images (background color = mean color of the original surroundings of the map content)
- a mask indicating the region of interest, excluding any non-map content: `*-MASK_content.png`
  *Values > 255 in the gray PNG image indicate map content area.*

Expected output is a rasterized version of the manually-created vector data for a subset of all possible closed shapes in the map.
Each edge is represented by a 1-pixel-large edge with value 255 in each `*-EDGE_target.png` file.


## Generation process
Vector data was annotated using [QGIS](https://www.qgis.org/), and rasterized using the embedded [GDAL](https://gdal.org/) tools.
The generation of this dataset required approximately 150 hours of manual work from the authors.
